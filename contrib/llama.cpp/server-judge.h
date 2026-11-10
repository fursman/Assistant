#pragma once

// Prefill-only typed-question judgment inside llama-server ("System One" style).
//
// The judge owns a few extra sequence ids in the SAME llama_context as the completion
// slots (started with --judge-slots N, which forces a unified KV cache). A request
// carries a shared context plus N typed questions. The context is prefilled once on the
// judge's own sequence (or copied from an idle slot whose cached prompt shares it), every
// question suffix is decoded as its own sequence in one batched llama_decode, and the
// logits at each answer position are read: no token is ever sampled or generated.
//
// Runs on the server's main loop thread between update_slots() iterations, so it never
// races the slots' own decode.
//
// v3 (judge-perf 2026-11-07), each part can be switched off for ablation (env LLAMA_JUDGE_*
// sets the server default, a request's "judge_opts" object overrides it for that request):
//   fast_tok   render the prompt once with a placeholder and tokenize the shared head once,
//              each question's tail separately (validated against a full tokenization)
//   cut_ctx    the cached prefix ends where the context ends, so the same context with other
//              questions, or a context that extends the cached one (a streaming transcript),
//              reuses it; the separator moves into the question suffixes
//   sys_ckpt   hybrid/recurrent models cannot rewind: keep the recurrent state at the end of
//              the system prompt (host or device copy) and restore it when only the context changed
//   pad        hybrid models split a multi-sequence batch into one ubatch per distinct length:
//              "full" pads every suffix of a round to the longest, "none" never pads,
//              "bucket" picks the padding targets with a cost model (c0 ms per ubatch, c1 ms per token)
//   trie       questions whose suffixes start with the same tokens decode that shared part once
//              (one extra forward pass, used only when the cost model says it pays)
//   sort       when there are more questions than slots, rounds take questions of similar length

#include "llama.h"
#include "common.h"
#include "chat.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

struct server_judge {
    using json = nlohmann::ordered_json;

    // llama_batch helpers (common_batch_add/clear left common when it moved to llama_batch_ext)
    static void batch_clear(llama_batch & b) { b.n_tokens = 0; }
    static void batch_add(llama_batch & b, llama_token id, llama_pos pos, llama_seq_id seq, bool logits) {
        b.token[b.n_tokens] = id; b.pos[b.n_tokens] = pos; b.n_seq_id[b.n_tokens] = 1;
        b.seq_id[b.n_tokens][0] = seq; b.logits[b.n_tokens] = logits; b.n_tokens++;
    }

    struct slot_cache { llama_seq_id seq; llama_tokens tokens; };
    using slot_cache_fn = std::function<std::vector<slot_cache>()>;

    // ------------------------------------------------------------------ options
    enum pad_mode { PAD_NONE = 0, PAD_FULL = 1, PAD_BUCKET = 2 };

    struct options {
        bool   fast_tok  = true;
        bool   cut_ctx   = true;
        bool   sys_ckpt  = true;
        bool   ckpt_dev  = true;    // keep the checkpoint in device memory (else host RAM)
        int    pad       = PAD_BUCKET;
        bool   trie      = false;   // measured: no gain on router/MultiRC at 2-6 slots, lexicographic order costs padding
        bool   sort      = true;
        double c0        = 62.0;    // ms per forward pass (ubatch) - measured on the target GPU
        double c1        = 0.75;    // ms per token
        int    min_ckpt  = 24;      // smallest system prefix worth a checkpoint (tokens)

        static bool env_bool(const char * name, bool def) {
            const char * v = getenv(name);
            if (!v || !*v) return def;
            return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
        }
        static double env_num(const char * name, double def) {
            const char * v = getenv(name);
            return (v && *v) ? atof(v) : def;
        }
        static int parse_pad(const std::string & s, int def) {
            if (s == "none") return PAD_NONE;
            if (s == "full") return PAD_FULL;
            if (s == "bucket") return PAD_BUCKET;
            return def;
        }
        static const char * pad_name(int p) { return p == PAD_NONE ? "none" : p == PAD_FULL ? "full" : "bucket"; }

        void from_env() {
            fast_tok = env_bool("LLAMA_JUDGE_FAST_TOK", fast_tok);
            cut_ctx  = env_bool("LLAMA_JUDGE_CUT_CTX",  cut_ctx);
            sys_ckpt = env_bool("LLAMA_JUDGE_SYS_CKPT", sys_ckpt);
            ckpt_dev = env_bool("LLAMA_JUDGE_CKPT_DEV", ckpt_dev);
            trie     = env_bool("LLAMA_JUDGE_TRIE",     trie);
            sort     = env_bool("LLAMA_JUDGE_SORT",     sort);
            c0       = env_num ("LLAMA_JUDGE_C0",       c0);
            c1       = env_num ("LLAMA_JUDGE_C1",       c1);
            min_ckpt = (int) env_num("LLAMA_JUDGE_MIN_CKPT", min_ckpt);
            const char * p = getenv("LLAMA_JUDGE_PAD");
            if (p) pad = parse_pad(p, pad);
        }
        void apply(const json & j) {
            if (!j.is_object()) return;
            fast_tok = j.value("fast_tok", fast_tok);
            cut_ctx  = j.value("cut_ctx",  cut_ctx);
            sys_ckpt = j.value("sys_ckpt", sys_ckpt);
            ckpt_dev = j.value("ckpt_dev", ckpt_dev);
            trie     = j.value("trie",     trie);
            sort     = j.value("sort",     sort);
            c0       = j.value("c0",       c0);
            c1       = j.value("c1",       c1);
            min_ckpt = j.value("min_ckpt", min_ckpt);
            if (j.contains("pad")) pad = parse_pad(j.at("pad").get<std::string>(), pad);
        }
        json to_json() const {
            return { {"fast_tok", fast_tok}, {"cut_ctx", cut_ctx}, {"sys_ckpt", sys_ckpt}, {"ckpt_dev", ckpt_dev},
                     {"pad", pad_name(pad)}, {"trie", trie}, {"sort", sort}, {"c0", c0}, {"c1", c1}, {"min_ckpt", min_ckpt} };
        }
    };

    llama_context * ctx = nullptr;
    const llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    const common_chat_templates * tmpls = nullptr;
    llama_memory_t mem = nullptr;

    llama_seq_id seq0 = -1;      // judge context sequence
    int n_qslots = 0;            // question sequences: seq0+1 .. seq0+n_qslots
    int n_batch = 0;
    int n_ubatch = 0;
    int n_ctx = 0;
    int n_vocab = 0;
    bool add_bos = false;
    bool hybrid = false;
    llama_token pad_token = 0;
    llama_batch batch{};
    llama_tokens cached;         // tokens resident on seq0
    std::string alias;
    options defaults;

    // recurrent-state checkpoint of seq0 at the end of a system prefix
    struct checkpoint {
        llama_tokens tokens;     // the prefix whose state is kept
        std::vector<uint8_t> data;
        bool on_device = false;
        bool valid() const { return !tokens.empty() && !data.empty(); }
        void clear() { tokens.clear(); data.clear(); data.shrink_to_fit(); }
    } ckpt;
    std::deque<size_t> sys_seen; // hashes of recent system prefixes (a checkpoint is made on the 2nd sighting)

    ~server_judge() { if (batch.token) llama_batch_free(batch); }

    int max_outputs = 0;         // logits rows one llama_decode may produce (server output buffer)

    void init(llama_context * c, const common_chat_templates * t, llama_seq_id first_seq, int qslots, const std::string & name, int n_outputs_max) {
        ctx = c; tmpls = t; seq0 = first_seq; n_qslots = qslots; alias = name; max_outputs = n_outputs_max;
        model = llama_get_model(ctx);
        vocab = llama_model_get_vocab(model);
        mem = llama_get_memory(ctx);
        n_batch = (int) llama_n_batch(ctx);
        n_ubatch = (int) llama_n_ubatch(ctx);
        n_ctx = (int) llama_n_ctx(ctx);
        n_vocab = llama_vocab_n_tokens(vocab);
        add_bos = llama_vocab_get_add_bos(vocab);
        hybrid = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        pad_token = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : 0;
        if (batch.token) llama_batch_free(batch);
        batch = llama_batch_init(std::max(n_batch, 1), 0, 1);
        cached.clear();
        ckpt.clear();
        sys_seen.clear();
        defaults = options();
        defaults.from_env();
    }

    bool enabled() const { return ctx != nullptr && n_qslots > 0; }

    // the server cleared or reloaded its memory: nothing is resident any more
    void invalidate() { cached.clear(); ckpt.clear(); }

    static double now_ms() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // ------------------------------------------------------------------ prompts
    std::string render(const std::string & system, const json & messages, const std::string & user, bool enable_thinking) {
        common_chat_templates_inputs in;
        if (!system.empty()) { common_chat_msg m; m.role = "system"; m.content = system; in.messages.push_back(m); }
        if (messages.is_array()) {
            for (const auto & msg : messages) {
                common_chat_msg m;
                m.role = msg.value("role", "user");
                const auto & c = msg.at("content");
                m.content = c.is_string() ? c.get<std::string>() : c.dump();
                in.messages.push_back(m);
            }
        }
        common_chat_msg u; u.role = "user"; u.content = user; in.messages.push_back(u);
        in.add_generation_prompt = true;
        in.enable_thinking = enable_thinking;
        in.use_jinja = true;
        return common_chat_templates_apply(tmpls, in).prompt.str();
    }

    llama_tokens tokenize_prompt(const std::string & text) {
        auto t = common_tokenize(vocab, text, false, true);
        if (add_bos && (t.empty() || t[0] != llama_vocab_bos(vocab))) t.insert(t.begin(), llama_vocab_bos(vocab));
        return t;
    }

    std::string piece(llama_token t) { return common_token_to_piece(vocab, t, true); }

    // number of leading tokens of `toks` whose text ends at or before byte `limit` of the prompt
    // text (pieces concatenate to the prompt for byte-level BPE); -1 if the pieces do not add up
    long tokens_before(const llama_tokens & toks, size_t limit, size_t text_size) {
        size_t off = 0; long k = 0, best = 0;
        const llama_token bos = llama_vocab_bos(vocab);
        bool bos_inserted = false;
        for (size_t i = 0; i < toks.size(); i++) {
            if (i == 0 && add_bos && toks[0] == bos) { bos_inserted = true; k++; best = k; continue; }
            off += piece(toks[i]).size();
            k++;
            if (off <= limit) best = k;
            if (off >= text_size) break;
        }
        (void) bos_inserted;
        if (off != text_size) return -1;
        return best;
    }

    // ------------------------------------------------------------------ options
    struct option_spec { std::string text; llama_tokens first_tokens; llama_tokens seq_tokens; bool partial = false; };

    static std::string lower(std::string s) { for (auto & c : s) c = tolower((unsigned char) c); return s; }
    static std::string upper(std::string s) { for (auto & c : s) c = toupper((unsigned char) c); return s; }
    static std::string cap(std::string s) { s = lower(s); if (!s.empty()) s[0] = toupper((unsigned char) s[0]); return s; }

    option_spec resolve_option(const json & o, const std::string & option_prefix) {
        option_spec spec;
        std::vector<std::string> variants;
        if (o.is_string()) {
            spec.text = o.get<std::string>();
            if (spec.text.size() == 1) {
                variants = { spec.text, " " + spec.text };   // never case-fold "A" into the article "a"
            } else {
                for (auto & base : { spec.text, lower(spec.text), cap(spec.text), upper(spec.text) }) {
                    variants.push_back(base); variants.push_back(" " + base);
                }
            }
        } else {
            spec.text = o.at("text").get<std::string>();
            for (auto & v : o.at("variants")) variants.push_back(v.get<std::string>());
        }
        std::unordered_set<llama_token> seen;
        for (auto & v : variants) {
            auto t = common_tokenize(vocab, v, false, false);
            if (t.size() == 1 && !seen.count(t[0])) { seen.insert(t[0]); spec.first_tokens.push_back(t[0]); }
        }
        spec.seq_tokens = common_tokenize(vocab, option_prefix + spec.text, false, false);
        if (spec.first_tokens.empty()) {
            spec.partial = true;
            if (!spec.seq_tokens.empty()) spec.first_tokens.push_back(spec.seq_tokens[0]);
        }
        return spec;
    }

    // ------------------------------------------------------------------ decoding
    struct stats_t {
        int n_decode = 0;          // llama_decode calls
        int n_rounds = 0;
        long n_real = 0;           // question tokens that carry text
        long n_padded = 0;         // pad tokens
        long n_trie_saved = 0;     // tokens not decoded thanks to shared question prefixes
        int n_trie_rounds = 0;
        double decode_ms = 0, logits_ms = 0, ckpt_ms = 0;
        std::string ckpt_event;    // "", "saved", "restored"
        std::vector<std::pair<int, double>> decodes; // (n_tokens, ms) per llama_decode
    };

    int decode_timed(stats_t & st) {
        const double t = now_ms();
        int rc = llama_decode(ctx, batch);
        if (rc == 0) llama_synchronize(ctx); // decode is asynchronous: wait, so the timings mean GPU time
        const double dt = now_ms() - t;
        st.n_decode++; st.decode_ms += dt;
        st.decodes.push_back({ batch.n_tokens, dt });
        return rc;
    }

    int decode_plain(const llama_tokens & toks, size_t from, size_t to, llama_seq_id seq, stats_t & st) {
        for (size_t i = from; i < to; i += n_batch) {
            batch_clear(batch);
            size_t end = std::min(to, i + (size_t) n_batch);
            for (size_t j = i; j < end; j++) batch_add(batch, toks[j], (llama_pos) j, seq, false);
            int rc = decode_timed(st);
            if (rc != 0) return rc;
        }
        return 0;
    }

    static size_t lcp(const llama_tokens & a, const llama_tokens & b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) n++;
        return n;
    }

    bool ckpt_save(size_t n, const llama_tokens & prefix, bool on_device, stats_t & st) {
        const double t = now_ms();
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | (on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : 0);
        const size_t sz = llama_state_seq_get_size_ext(ctx, seq0, fl);
        if (sz == 0) { ckpt.clear(); return false; }
        ckpt.data.resize(sz);
        const size_t got = llama_state_seq_get_data_ext(ctx, ckpt.data.data(), sz, seq0, fl);
        if (got != sz) { ckpt.clear(); return false; }
        ckpt.tokens.assign(prefix.begin(), prefix.begin() + n);
        ckpt.on_device = on_device;
        st.ckpt_ms += now_ms() - t;
        st.ckpt_event = "saved";
        return true;
    }

    bool ckpt_restore(stats_t & st) {
        const double t = now_ms();
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | (ckpt.on_device ? LLAMA_STATE_SEQ_FLAGS_ON_DEVICE : 0);
        const size_t got = llama_state_seq_set_data_ext(ctx, ckpt.data.data(), ckpt.data.size(), seq0, fl);
        if (got != ckpt.data.size()) { ckpt.clear(); return false; }
        // the recurrent state now ends at the checkpoint, so the attention cells past it can go
        if (!llama_memory_seq_rm(mem, seq0, (llama_pos) ckpt.tokens.size(), -1)) { ckpt.clear(); return false; }
        st.ckpt_ms += now_ms() - t;
        st.ckpt_event = "restored";
        return true;
    }

    static size_t hash_tokens(const llama_tokens & t, size_t n) {
        size_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n && i < t.size(); i++) { h ^= (size_t) (uint32_t) t[i]; h *= 1099511628211ull; }
        return h ^ n;
    }

    // Make seq0 hold exactly `prefix`. Sources, best first: seq0's own cache (extend, trim, or
    // restore the system checkpoint), an idle completion slot whose cached prompt shares a longer
    // prefix (copy + trim), or a fresh prefill. `sys_end` is where the system part of the prompt
    // ends (0 = unknown). Returns tokens computed, or -1 on failure; `source` says what was used.
    int ensure_prefix(const llama_tokens & prefix, size_t sys_end, const options & opt, const std::vector<slot_cache> & slots,
                      std::string & source, std::string & err, stats_t & st) {
        size_t own = lcp(cached, prefix);
        // trimming is only possible on attention-only memory (or within MTP rollback snapshots)
        auto trim_to = [&](llama_seq_id seq, size_t keep, size_t have) -> bool {
            if (have == keep) return true;
            if (keep == 0) return false;
            return llama_memory_seq_rm(mem, seq, (llama_pos) keep, -1);
        };
        const bool ck_ok = opt.sys_ckpt && hybrid && ckpt.valid() && ckpt.tokens.size() <= prefix.size()
                           && lcp(ckpt.tokens, prefix) == ckpt.tokens.size();
        // what seq0 can offer without a fresh prefill
        size_t own_usable = own;
        if (own < cached.size() && hybrid) {
            own_usable = (ck_ok && own >= ckpt.tokens.size()) ? ckpt.tokens.size() : 0;
        }
        // candidate from slots: longest shared prefix that beats what seq0 already has
        const slot_cache * best = nullptr; size_t best_n = own_usable;
        for (const auto & s : slots) {
            size_t n = lcp(s.tokens, prefix);
            if (n > best_n + 16) { best = &s; best_n = n; }
        }
        size_t common = 0;
        if (best) {
            llama_memory_seq_rm(mem, seq0, -1, -1);
            llama_memory_seq_cp(mem, best->seq, seq0, -1, -1);
            if (trim_to(seq0, best_n, best->tokens.size())) {
                cached.assign(prefix.begin(), prefix.begin() + best_n);
                common = best_n;
                source = "slot";
            } else {
                llama_memory_seq_rm(mem, seq0, -1, -1); cached.clear(); common = 0; source = "fresh";
            }
        } else if (own < cached.size()) {
            if (trim_to(seq0, own, cached.size())) { cached.resize(own); common = own; source = own ? "cache" : "fresh"; }
            else if (ck_ok && own >= ckpt.tokens.size() && ckpt_restore(st)) {
                cached.resize(ckpt.tokens.size()); common = cached.size(); source = "ckpt";
            } else { llama_memory_seq_rm(mem, seq0, -1, -1); cached.clear(); common = 0; source = "fresh"; }
        } else {
            common = own; source = own ? "cache" : "fresh";
        }
        // a system prefix seen before (and not in the checkpoint yet): stop the prefill at its end once
        // and keep the state there, so the next context change restores it instead of starting over
        if (opt.sys_ckpt && hybrid && sys_end >= (size_t) opt.min_ckpt && common < sys_end && sys_end < prefix.size()) {
            const size_t h = hash_tokens(prefix, sys_end);
            const bool seen = std::find(sys_seen.begin(), sys_seen.end(), h) != sys_seen.end();
            const bool have = ckpt.valid() && ckpt.tokens.size() == sys_end && lcp(ckpt.tokens, prefix) == sys_end;
            if (seen && !have) {
                int rc = decode_plain(prefix, common, sys_end, seq0, st);
                if (rc != 0) {
                    err = "llama_decode failed on prefix (rc=" + std::to_string(rc) + ")";
                    llama_memory_seq_rm(mem, seq0, -1, -1); cached.clear();
                    return -1;
                }
                cached.assign(prefix.begin(), prefix.begin() + sys_end);
                const size_t n_done = sys_end - common;
                common = sys_end;
                ckpt_save(sys_end, prefix, opt.ckpt_dev, st);
                int rc2 = decode_plain(prefix, common, prefix.size(), seq0, st);
                if (rc2 != 0) {
                    err = "llama_decode failed on prefix (rc=" + std::to_string(rc2) + ")";
                    llama_memory_seq_rm(mem, seq0, -1, -1); cached.clear();
                    return -1;
                }
                cached = prefix;
                return (int) (n_done + prefix.size() - sys_end);
            }
        }
        if (sys_end > 0 && sys_end < prefix.size()) {
            const size_t h = hash_tokens(prefix, sys_end);
            sys_seen.push_back(h);
            while (sys_seen.size() > 8) sys_seen.pop_front();
        }
        if (common < prefix.size()) {
            int rc = decode_plain(prefix, common, prefix.size(), seq0, st);
            if (rc != 0) {
                err = "llama_decode failed on prefix (rc=" + std::to_string(rc) + ")";
                llama_memory_seq_rm(mem, seq0, -1, -1); cached.clear();
                return -1;
            }
            cached = prefix;
        }
        return (int) (prefix.size() - common);
    }

    struct row_stats { float mx; double lse; };
    row_stats row_lse(const float * logits) {
        float mx = logits[0];
        for (int i = 1; i < n_vocab; i++) mx = std::max(mx, logits[i]);
        // float terms in blocks, double accumulation across blocks: ~4x faster than double exp, ~1e-7 relative
        double s = 0;
        for (int i0 = 0; i0 < n_vocab; i0 += 1024) {
            const int i1 = std::min(n_vocab, i0 + 1024);
            float b = 0.0f;
            for (int i = i0; i < i1; i++) b += expf(logits[i] - mx);
            s += b;
        }
        return { mx, std::log(s) };
    }

    json topk_json(const float * logits, const row_stats & st, int k) {
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; i++) idx[i] = i;
        k = std::min(k, n_vocab);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        json out = json::array();
        for (int i = 0; i < k; i++) {
            double lp = (double) (logits[idx[i]] - st.mx) - st.lse;
            std::string p = piece(idx[i]);
            json pj;
            if (is_valid_utf8_str(p)) pj = p; else { pj = json::array(); for (unsigned char c : p) pj.push_back((int) c); }
            out.push_back({ {"token", idx[i]}, {"piece", pj}, {"prob", std::exp(lp)} });
        }
        return out;
    }

    static bool is_valid_utf8_str(const std::string & s) {
        size_t i = 0, n = s.size();
        while (i < n) {
            unsigned char c = s[i];
            int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
            if (!len || i + len > n) return false;
            for (int k = 1; k < len; k++) if ((s[i + k] & 0xC0) != 0x80) return false;
            i += len;
        }
        return true;
    }

    struct seq_job {
        int q_idx; int opt_idx;
        llama_tokens tokens; std::vector<int> logit_pos;
        std::vector<double> opt_logprobs; double mass = 0; json top; std::vector<double> tok_logprobs;
    };

    // cost of one batched step of n_tok tokens in n_seq equal-length sequences (ubatch splits included)
    double step_cost(const options & opt, long n_tok, long n_seq) const {
        if (n_tok <= 0) return 0;
        long per_ub = std::max(1L, (long) n_ubatch / std::max(1L, n_seq)) * std::max(1L, n_seq);
        long n_ub = (n_tok + per_ub - 1) / per_ub;
        return opt.c0 * n_ub + opt.c1 * n_tok;
    }

    // padding targets for lengths `L` (any order): returns the padded length of each job
    std::vector<size_t> pad_targets(const options & opt, const std::vector<size_t> & L) const {
        std::vector<size_t> out(L);
        if (L.empty()) return out;
        const size_t lmax = *std::max_element(L.begin(), L.end());
        if (!hybrid || opt.pad == PAD_NONE) return out;
        if (opt.pad == PAD_FULL) { std::fill(out.begin(), out.end(), lmax); return out; }
        // bucket: choose target levels among the distinct lengths; jobs pad up to the next level.
        // split_equal runs one step per level: level k decodes (t_k - t_{k-1}) tokens for every job
        // whose padded length is >= t_k.
        std::vector<size_t> d(L); std::sort(d.begin(), d.end()); d.erase(std::unique(d.begin(), d.end()), d.end());
        const size_t m = d.size();
        auto n_ge = [&](size_t v) { long c = 0; for (auto x : L) if (x >= v) c++; return c; };
        // dp[i] = best cost when level d[i] is a target and all lengths <= d[i] are covered
        // transition from previous target j (or start): jobs with length in (d[j], d[i]] pad to d[i]
        std::vector<double> dp(m, 1e300); std::vector<int> prv(m, -1);
        for (size_t i = 0; i < m; i++) {
            for (int j = -1; j < (int) i; j++) {
                const size_t lo = j < 0 ? 0 : d[j];
                // after the previous level, every job that is longer than d[j] still runs: count them
                const long n_run = j < 0 ? (long) L.size() : n_ge(d[j] + 1);
                const double c = (j < 0 ? 0 : dp[j]) + step_cost(opt, n_run * (long) (d[i] - lo), n_run);
                if (c < dp[i]) { dp[i] = c; prv[i] = j; }
            }
        }
        std::vector<size_t> levels;
        for (int i = (int) m - 1; i >= 0; i = prv[i]) levels.push_back(d[i]);
        std::sort(levels.begin(), levels.end());
        for (size_t k = 0; k < L.size(); k++) out[k] = *std::lower_bound(levels.begin(), levels.end(), L[k]);
        return out;
    }

    double plan_cost(const options & opt, const std::vector<size_t> & padded) const {
        std::vector<size_t> d(padded); std::sort(d.begin(), d.end()); d.erase(std::unique(d.begin(), d.end()), d.end());
        double c = 0; size_t lo = 0;
        for (size_t v : d) {
            long n = 0; for (auto x : padded) if (x >= v) n++;
            c += step_cost(opt, n * (long) (v - lo), n);
            lo = v;
        }
        return c;
    }

    bool run_jobs(std::vector<seq_job> & jobs, const std::vector<std::vector<option_spec>> & opts, int top_k, const options & opt,
                  stats_t & st, std::string & err) {
        const llama_pos P = (llama_pos) cached.size();
        // order: with more jobs than slots, rounds take neighbours. The trie wants shared prefixes
        // adjacent (lexicographic), padding wants similar lengths adjacent.
        std::vector<size_t> order(jobs.size());
        std::iota(order.begin(), order.end(), 0);
        if (opt.sort && (int) jobs.size() > n_qslots) {
            if (opt.trie) {
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return jobs[a].tokens < jobs[b].tokens; });
            } else {
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return jobs[a].tokens.size() < jobs[b].tokens.size(); });
            }
        }
        auto qseq = [&](size_t k) { return seq0 + 1 + (llama_seq_id) k; };
        size_t r0 = 0;
        while (r0 < order.size()) {
            // form a round: <= n_qslots jobs within the KV budget (padding counted on hybrids)
            size_t r1 = r0, round_tokens = 0, L_max = 0;
            while (r1 < order.size() && (int) (r1 - r0) < n_qslots) {
                const size_t L = jobs[order[r1]].tokens.size();
                const size_t new_max = std::max(L_max, L);
                const size_t cost = hybrid ? (r1 - r0 + 1) * new_max : round_tokens + L;
                if (r1 > r0 && (size_t) P + cost > (size_t) n_ctx) break;
                round_tokens += L; L_max = new_max; r1++;
            }
            if (r1 == r0) { err = "context too long for the KV budget"; return false; }
            const size_t n = r1 - r0;
            st.n_rounds++;
            auto cleanup = [&]() { for (size_t k = 0; k < n; k++) llama_memory_seq_rm(mem, qseq(k), -1, -1); };
            for (size_t k = 0; k < n; k++) llama_memory_seq_cp(mem, seq0, qseq(k), -1, -1);

            std::vector<size_t> L(n);
            for (size_t k = 0; k < n; k++) L[k] = jobs[order[r0 + k]].tokens.size();

            // ---- trie: one shared depth s for the round; groups = jobs whose first s tokens agree
            size_t s_best = 0; std::vector<int> group_of(n, -1); std::vector<size_t> leaders;
            std::vector<size_t> padded_best = pad_targets(opt, L);
            if (opt.trie && n > 1) {
                const double base = plan_cost(opt, padded_best);
                size_t min_len = *std::min_element(L.begin(), L.end());
                // candidate depths: the pairwise common prefix lengths, capped below the shortest job
                std::vector<size_t> cand;
                for (size_t a = 0; a < n; a++) for (size_t b = a + 1; b < n; b++) {
                    size_t c = lcp(jobs[order[r0 + a]].tokens, jobs[order[r0 + b]].tokens);
                    c = std::min(c, min_len - 1);
                    if (c > 0) cand.push_back(c);
                }
                std::sort(cand.begin(), cand.end()); cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                double best_gain = 0;
                for (size_t s : cand) {
                    // group by the first s tokens
                    std::map<llama_tokens, int> gid; std::vector<int> g(n); int ng = 0;
                    for (size_t k = 0; k < n; k++) {
                        const auto & t = jobs[order[r0 + k]].tokens;
                        llama_tokens key(t.begin(), t.begin() + s);
                        auto it = gid.find(key);
                        if (it == gid.end()) { gid[key] = ng; g[k] = ng++; } else g[k] = it->second;
                    }
                    if (ng == (int) n) continue;
                    std::vector<size_t> rest(n);
                    for (size_t k = 0; k < n; k++) rest[k] = L[k] - s;
                    auto pr = pad_targets(opt, rest);
                    const double c = step_cost(opt, (long) ng * (long) s, ng) + plan_cost(opt, pr);
                    const double gain = base - c;
                    if (gain > best_gain) { best_gain = gain; s_best = s; group_of = g; }
                }
                if (s_best > 0) {
                    std::vector<size_t> rest(n);
                    for (size_t k = 0; k < n; k++) rest[k] = L[k] - s_best;
                    padded_best = pad_targets(opt, rest);
                    for (auto & v : padded_best) v += s_best;
                    int ng = *std::max_element(group_of.begin(), group_of.end()) + 1;
                    leaders.assign(ng, SIZE_MAX);
                    for (size_t k = 0; k < n; k++) if (leaders[group_of[k]] == SIZE_MAX) leaders[group_of[k]] = k;
                }
            }

            // ---- step A (trie): each group's leader decodes the shared s tokens, members copy it
            if (s_best > 0) {
                st.n_trie_rounds++;
                std::vector<std::pair<size_t, size_t>> ents; // (k, ti)
                for (size_t gi = 0; gi < leaders.size(); gi++)
                    for (size_t t = 0; t < s_best; t++) ents.push_back({ leaders[gi], t });
                for (size_t e0 = 0; e0 < ents.size(); e0 += n_batch) {
                    batch_clear(batch);
                    size_t e1 = std::min(ents.size(), e0 + (size_t) n_batch);
                    for (size_t e = e0; e < e1; e++) {
                        auto & jb = jobs[order[r0 + ents[e].first]];
                        batch_add(batch, jb.tokens[ents[e].second], P + (llama_pos) ents[e].second, qseq(ents[e].first), false);
                    }
                    int rc = decode_timed(st);
                    if (rc != 0) { err = "llama_decode failed on shared question prefixes (rc=" + std::to_string(rc) + ")"; cleanup(); return false; }
                }
                for (size_t k = 0; k < n; k++) {
                    const size_t lead = leaders[group_of[k]];
                    if (lead == k) continue;
                    llama_memory_seq_rm(mem, qseq(k), -1, -1);
                    llama_memory_seq_cp(mem, qseq(lead), qseq(k), -1, -1);
                    st.n_trie_saved += (long) s_best;
                }
                st.n_real += (long) leaders.size() * (long) s_best;
            }

            // ---- step B: every job decodes the rest of its suffix (padded per plan), logits at the answer
            const size_t t_start = s_best;
            // column-major chunks keep the per-sequence lengths equal inside each llama_decode on hybrids
            std::vector<std::pair<size_t, size_t>> ents; // (k, ti)
            size_t Lp_max = 0;
            for (size_t k = 0; k < n; k++) Lp_max = std::max(Lp_max, padded_best[k]);
            if (hybrid) {
                for (size_t t = t_start; t < Lp_max; t++)
                    for (size_t k = 0; k < n; k++) if (t < padded_best[k]) ents.push_back({ k, t });
            } else {
                for (size_t k = 0; k < n; k++)
                    for (size_t t = t_start; t < L[k]; t++) ents.push_back({ k, t });
            }
            for (size_t k = 0; k < n; k++) {
                st.n_real += (long) (L[k] - t_start);
                if (hybrid) st.n_padded += (long) (padded_best[k] - L[k]);
            }
            for (size_t e0 = 0; e0 < ents.size();) {
                size_t e1 = std::min(ents.size(), e0 + (size_t) n_batch);
                batch_clear(batch);
                std::vector<std::pair<int,int>> want; // (batch row, logit index)
                std::vector<size_t> want_job;
                size_t e = e0;
                for (; e < e1; e++) {
                    const size_t k = ents[e].first; const int ti = (int) ents[e].second;
                    auto & jb = jobs[order[r0 + k]];
                    const bool is_pad = ti >= (int) jb.tokens.size();
                    auto it = is_pad ? jb.logit_pos.end() : std::find(jb.logit_pos.begin(), jb.logit_pos.end(), ti);
                    bool want_logits = it != jb.logit_pos.end();
                    // a chunk may not read more logits rows than the context's output buffer holds
                    if (want_logits && max_outputs > 0 && (int) want.size() >= max_outputs && batch.n_tokens > 0) break;
                    if (want_logits) { want.push_back({ batch.n_tokens, (int) (it - jb.logit_pos.begin()) }); want_job.push_back(order[r0 + k]); }
                    batch_add(batch, is_pad ? pad_token : jb.tokens[ti], P + ti, qseq(k), want_logits);
                }
                e1 = e;
                int rc = decode_timed(st);
                if (rc != 0) { err = "llama_decode failed on questions (rc=" + std::to_string(rc) + ")"; cleanup(); return false; }
                const double tl = now_ms();
                for (size_t w = 0; w < want.size(); w++) {
                    auto & job = jobs[want_job[w]];
                    const float * logits = llama_get_logits_ith(ctx, want[w].first);
                    if (!logits) { err = "null logits"; cleanup(); return false; }
                    row_stats rs = row_lse(logits);
                    auto lp = [&](llama_token t) { return (double) (logits[t] - rs.mx) - rs.lse; };
                    if (job.opt_idx < 0) {
                        const auto & qopts = opts[job.q_idx];
                        job.opt_logprobs.assign(qopts.size(), -INFINITY);
                        double mass = 0;
                        for (size_t o = 0; o < qopts.size(); o++) {
                            double acc = -INFINITY;
                            for (auto t : qopts[o].first_tokens) {
                                double v = lp(t);
                                acc = acc == -INFINITY ? v : std::max(acc, v) + std::log1p(std::exp(-std::fabs(acc - v)));
                            }
                            job.opt_logprobs[o] = acc; mass += std::exp(acc);
                        }
                        job.mass = mass;
                        if (top_k > 0) job.top = topk_json(logits, rs, top_k);
                    } else {
                        const auto & spec = opts[job.q_idx][job.opt_idx];
                        if (job.tok_logprobs.empty()) job.tok_logprobs.assign(spec.seq_tokens.size(), 0.0);
                        job.tok_logprobs[want[w].second] = lp(spec.seq_tokens[want[w].second]);
                    }
                }
                st.logits_ms += now_ms() - tl;
                e0 = e1;
            }
            cleanup();
            r0 = r1;
        }
        return true;
    }

    // ------------------------------------------------------------------ request
    // {"system", "messages":[{role,content}...] (optional prior turns), "context", "separator",
    //  "enable_thinking", "assistant_prefix", "option_prefix", "scoring", "top_k", "questions":[{id,text,options[,scoring]}],
    //  "judge_opts": {...} (optional per-request overrides of the server's judge options), "debug": bool}
    json judge(const json & req, const slot_cache_fn & slot_caches) {
        const double t0 = now_ms();
        options opt = defaults;
        if (req.contains("judge_opts")) opt.apply(req.at("judge_opts"));
        const bool debug = req.value("debug", false);
        const bool enable_thinking = req.value("enable_thinking", false);
        const std::string assistant_prefix = req.value("assistant_prefix", "");
        const std::string option_prefix = req.value("option_prefix", "");
        const std::string scoring_default = req.value("scoring", "first_token");
        const int top_k = req.value("top_k", 0);
        const auto & qs = req.at("questions");
        if (!qs.is_array() || qs.empty()) throw std::runtime_error("questions must be a non-empty array");
        const json messages = req.value("messages", json::array());
        const std::string system = req.value("system", "");
        const std::string context = req.value("context", "");
        const std::string separator = req.value("separator", "\n\n");

        std::vector<std::string> qtext(qs.size());
        for (size_t i = 0; i < qs.size(); i++) qtext[i] = qs[i].value("text", "");

        // ---- prompts: render once with a placeholder where the question goes
        static const std::string PH = "@@JUDGE_Q_7f3a9c@@";
        const std::string ph_text = render(system, messages, context + separator + PH, enable_thinking) + assistant_prefix;
        const size_t ph = ph_text.find(PH);
        const bool ph_ok = ph != std::string::npos && ph_text.find(PH, ph + 1) == std::string::npos;
        std::string pre, post;
        size_t ctx_begin = std::string::npos, ctx_end = std::string::npos;
        if (ph_ok) {
            pre = ph_text.substr(0, ph); post = ph_text.substr(ph + PH.size());
            if (pre.size() >= separator.size() + context.size()
                && pre.compare(pre.size() - separator.size(), separator.size(), separator) == 0
                && pre.compare(pre.size() - separator.size() - context.size(), context.size(), context) == 0) {
                ctx_end = pre.size() - separator.size();
                ctx_begin = ctx_end - context.size();
            }
        }
        std::vector<std::string> texts(qs.size());
        bool templ_ok = ph_ok;
        if (ph_ok) {
            for (size_t i = 0; i < qs.size(); i++) texts[i] = pre + qtext[i] + post;
            // the template must place the text verbatim: check with question 0 rendered for real
            const std::string real0 = render(system, messages, context + separator + qtext[0], enable_thinking) + assistant_prefix;
            if (real0 != texts[0]) { templ_ok = false; ctx_begin = ctx_end = std::string::npos; }
        }
        if (!templ_ok) {
            for (size_t i = 0; i < qs.size(); i++) texts[i] = render(system, messages, context + separator + qtext[i], enable_thinking) + assistant_prefix;
        }
        const double t_render = now_ms();

        // ---- tokens
        std::vector<llama_tokens> full(qs.size());
        bool fast = false;
        llama_tokens head;
        if (opt.fast_tok && templ_ok && !pre.empty() && pre.back() == '\n') {
            bool starts_ok = true;
            for (auto & q : qtext) if (q.empty() || isspace((unsigned char) q[0])) { starts_ok = false; break; }
            if (starts_ok) {
                head = tokenize_prompt(pre);
                for (size_t i = 0; i < qs.size(); i++) {
                    auto tail = common_tokenize(vocab, qtext[i] + post, false, true);
                    full[i] = head; full[i].insert(full[i].end(), tail.begin(), tail.end());
                }
                // validate the split against a full tokenization (first and last question)
                fast = full[0] == tokenize_prompt(texts[0]) && (qs.size() == 1 || full.back() == tokenize_prompt(texts.back()));
            }
        }
        if (!fast) {
            for (size_t i = 0; i < qs.size(); i++) full[i] = tokenize_prompt(texts[i]);
        }
        std::vector<std::vector<option_spec>> opts(qs.size());
        std::vector<std::string> scoring(qs.size());
        for (size_t i = 0; i < qs.size(); i++) {
            const auto & q = qs[i];
            if (full[i].empty()) throw std::runtime_error("empty prompt");
            for (auto & o : q.at("options")) opts[i].push_back(resolve_option(o, option_prefix));
            if (opts[i].empty()) throw std::runtime_error("question has no options");
            for (auto & s : opts[i]) if (s.seq_tokens.empty() || s.first_tokens.empty()) throw std::runtime_error("option '" + s.text + "' has no tokens");
            scoring[i] = q.value("scoring", scoring_default);
        }
        size_t min_len = full[0].size();
        for (auto & f : full) min_len = std::min(min_len, f.size());
        size_t P = 0;
        while (P + 1 < min_len) {
            llama_token t = full[0][P]; bool same = true;
            for (auto & f : full) if (f[P] != t) { same = false; break; }
            if (!same) break;
            P++;
        }
        // token positions of the context's end and of the system part's end
        size_t sys_end = 0;
        size_t P_cut = P;
        if (ctx_end != std::string::npos) {
            const llama_tokens & ref = fast ? head : full[0];
            const size_t ref_size = fast ? pre.size() : texts[0].size();
            long k_end = tokens_before(ref, ctx_end, ref_size);
            long k_sys = tokens_before(ref, ctx_begin, ref_size);
            if (k_end > 0) P_cut = std::min(P, (size_t) k_end);
            if (k_sys > 0) sys_end = std::min((size_t) k_sys, P_cut);
        }
        if (opt.cut_ctx && P_cut > 0) P = P_cut;
        llama_tokens prefix(full[0].begin(), full[0].begin() + P);
        const double t1 = now_ms();

        std::string err, source;
        stats_t st;
        int n_new = ensure_prefix(prefix, sys_end, opt, slot_caches ? slot_caches() : std::vector<slot_cache>{}, source, err, st);
        if (n_new < 0) throw std::runtime_error(err);
        const double t2 = now_ms();
        const int n_decode_prefix = st.n_decode;

        std::vector<seq_job> jobs; size_t n_qtok = 0;
        for (size_t i = 0; i < qs.size(); i++) {
            llama_tokens suffix(full[i].begin() + P, full[i].end());
            if (scoring[i] == "sequence") {
                for (size_t o = 0; o < opts[i].size(); o++) {
                    seq_job jb; jb.q_idx = (int) i; jb.opt_idx = (int) o; jb.tokens = suffix;
                    const auto & stt = opts[i][o].seq_tokens;
                    for (size_t k = 0; k < stt.size(); k++) jb.logit_pos.push_back((int) (suffix.size() - 1 + k));
                    jb.tokens.insert(jb.tokens.end(), stt.begin(), stt.end());
                    jb.tokens.pop_back();
                    n_qtok += jb.tokens.size(); jobs.push_back(std::move(jb));
                }
            } else {
                seq_job jb; jb.q_idx = (int) i; jb.opt_idx = -1; jb.tokens = suffix;
                jb.logit_pos.push_back((int) suffix.size() - 1);
                n_qtok += jb.tokens.size(); jobs.push_back(std::move(jb));
            }
        }
        if (!run_jobs(jobs, opts, top_k, opt, st, err)) throw std::runtime_error(err);
        const double t3 = now_ms();

        json out;
        out["model"] = alias;
        out["n_prefix_tokens"] = P;
        out["n_prefix_computed"] = n_new;
        out["prefix_source"] = source;
        out["n_question_tokens"] = n_qtok;
        out["n_sequences"] = jobs.size();
        out["n_rounds"] = st.n_rounds;
        out["n_decode_calls"] = st.n_decode;
        out["n_decode_prefix"] = n_decode_prefix;
        out["n_padded_tokens"] = st.n_padded;
        out["n_trie_saved_tokens"] = st.n_trie_saved;
        out["n_system_tokens"] = sys_end;
        out["checkpoint"] = st.ckpt_event;
        out["fast_tokenize"] = fast;
        out["timings"] = { {"tokenize_ms", t1 - t0}, {"render_ms", t_render - t0}, {"prefill_ms", t2 - t1}, {"questions_ms", t3 - t2},
                           {"decode_ms", st.decode_ms}, {"logits_ms", st.logits_ms}, {"checkpoint_ms", st.ckpt_ms}, {"total_ms", now_ms() - t0} };
        if (debug) {
            out["judge_opts"] = opt.to_json();
            json d = json::array();
            for (auto & x : st.decodes) d.push_back({ x.first, x.second });
            out["decodes"] = d;
        }
        json jq = json::array();
        for (size_t i = 0; i < qs.size(); i++) {
            json r; r["id"] = qs[i].value("id", std::to_string(i));
            r["n_tokens"] = full[i].size() - P; r["scoring"] = scoring[i];
            json jo = json::array();
            if (scoring[i] == "sequence") {
                std::vector<double> sums;
                for (auto & jb : jobs) if (jb.q_idx == (int) i) { double s = 0; for (double v : jb.tok_logprobs) s += v; sums.push_back(s); }
                double mx = *std::max_element(sums.begin(), sums.end()), z = 0;
                for (double s : sums) z += std::exp(s - mx);
                size_t k = 0; int best = 0;
                for (auto & jb : jobs) if (jb.q_idx == (int) i) {
                    const auto & spec = opts[i][jb.opt_idx];
                    jo.push_back({ {"text", spec.text}, {"logprob", sums[k]}, {"prob", std::exp(sums[k] - mx) / z},
                                   {"n_tokens", spec.seq_tokens.size()}, {"token_logprobs", jb.tok_logprobs} });
                    if (sums[k] > sums[best]) best = (int) k;
                    k++;
                }
                r["argmax"] = opts[i][best].text;
            } else {
                const seq_job * jb = nullptr;
                for (auto & j : jobs) if (j.q_idx == (int) i) { jb = &j; break; }
                double mx = -INFINITY; for (double v : jb->opt_logprobs) mx = std::max(mx, v);
                double z = 0; for (double v : jb->opt_logprobs) z += std::exp(v - mx);
                int best = 0;
                for (size_t o = 0; o < opts[i].size(); o++) {
                    jo.push_back({ {"text", opts[i][o].text}, {"logprob", jb->opt_logprobs[o]}, {"prob", std::exp(jb->opt_logprobs[o] - mx) / z},
                                   {"tokens", opts[i][o].first_tokens}, {"partial", opts[i][o].partial} });
                    if (jb->opt_logprobs[o] > jb->opt_logprobs[best]) best = (int) o;
                }
                r["mass"] = jb->mass; r["argmax"] = opts[i][best].text;
                if (top_k > 0) r["top"] = jb->top;
            }
            r["options"] = jo;
            jq.push_back(r);
        }
        out["questions"] = jq;
        return out;
    }
};
