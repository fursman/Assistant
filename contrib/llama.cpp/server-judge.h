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

#include "llama.h"
#include "common.h"
#include "chat.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

struct server_judge {
    using json = nlohmann::ordered_json;

    struct slot_cache { llama_seq_id seq; llama_tokens tokens; };
    using slot_cache_fn = std::function<std::vector<slot_cache>()>;

    llama_context * ctx = nullptr;
    const llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    const common_chat_templates * tmpls = nullptr;
    llama_memory_t mem = nullptr;

    llama_seq_id seq0 = -1;      // judge context sequence
    int n_qslots = 0;            // question sequences: seq0+1 .. seq0+n_qslots
    int n_batch = 0;
    int n_ctx = 0;
    int n_vocab = 0;
    bool add_bos = false;
    bool hybrid = false;
    llama_token pad_token = 0;
    llama_batch batch{};
    llama_tokens cached;         // tokens resident on seq0
    std::string alias;

    ~server_judge() { if (batch.token) llama_batch_free(batch); }

    int max_outputs = 0;         // logits rows one llama_decode may produce (server output buffer)

    void init(llama_context * c, const common_chat_templates * t, llama_seq_id first_seq, int qslots, const std::string & name, int n_outputs_max) {
        ctx = c; tmpls = t; seq0 = first_seq; n_qslots = qslots; alias = name; max_outputs = n_outputs_max;
        model = llama_get_model(ctx);
        vocab = llama_model_get_vocab(model);
        mem = llama_get_memory(ctx);
        n_batch = (int) llama_n_batch(ctx);
        n_ctx = (int) llama_n_ctx(ctx);
        n_vocab = llama_vocab_n_tokens(vocab);
        add_bos = llama_vocab_get_add_bos(vocab);
        hybrid = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        pad_token = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : 0;
        if (batch.token) llama_batch_free(batch);
        batch = llama_batch_init(n_batch, 0, 1);
        cached.clear();
    }

    bool enabled() const { return ctx != nullptr && n_qslots > 0; }

    // the server cleared or reloaded its memory: nothing is resident any more
    void invalidate() { cached.clear(); }

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
        return common_chat_templates_apply(tmpls, in).prompt;
    }

    llama_tokens tokenize_prompt(const std::string & text) {
        auto t = common_tokenize(vocab, text, false, true);
        if (add_bos && (t.empty() || t[0] != llama_vocab_bos(vocab))) t.insert(t.begin(), llama_vocab_bos(vocab));
        return t;
    }

    std::string piece(llama_token t) { return common_token_to_piece(vocab, t, true); }

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
    int decode_plain(const llama_tokens & toks, llama_seq_id seq, llama_pos pos0) {
        for (size_t i = 0; i < toks.size(); i += n_batch) {
            common_batch_clear(batch);
            size_t end = std::min(toks.size(), i + (size_t) n_batch);
            for (size_t j = i; j < end; j++) common_batch_add(batch, toks[j], pos0 + (llama_pos) j, { seq }, false);
            int rc = llama_decode(ctx, batch);
            if (rc != 0) return rc;
        }
        return 0;
    }

    static size_t lcp(const llama_tokens & a, const llama_tokens & b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) n++;
        return n;
    }

    // Make seq0 hold exactly `prefix`. Sources, best first: seq0's own cache (extend or trim),
    // an idle completion slot whose cached prompt shares a longer prefix (copy + trim), or a
    // fresh prefill. Returns tokens computed, or -1 on failure; `source` says what was used.
    int ensure_prefix(const llama_tokens & prefix, const std::vector<slot_cache> & slots, std::string & source, std::string & err) {
        size_t own = lcp(cached, prefix);
        // trimming is only possible on attention-only memory; try it, else the cache is unusable beyond `own`
        auto trim_to = [&](llama_seq_id seq, size_t keep, size_t have) -> bool {
            if (have == keep) return true;
            if (keep == 0) return false;
            return llama_memory_seq_rm(mem, seq, (llama_pos) keep, -1);
        };
        // candidate from slots: longest shared prefix that beats what seq0 already has
        const slot_cache * best = nullptr; size_t best_n = own;
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
            else { llama_memory_seq_rm(mem, seq0, -1, -1); cached.clear(); common = 0; source = "fresh"; }
        } else {
            common = own; source = own == prefix.size() ? "cache" : (own ? "cache" : "fresh");
        }
        llama_tokens rest(prefix.begin() + common, prefix.end());
        if (!rest.empty()) {
            int rc = decode_plain(rest, seq0, (llama_pos) common);
            if (rc != 0) {
                err = "llama_decode failed on prefix (rc=" + std::to_string(rc) + ")";
                llama_memory_seq_rm(mem, seq0, -1, -1); cached.clear();
                return -1;
            }
            cached = prefix;
        }
        return (int) rest.size();
    }

    struct row_stats { float mx; double lse; };
    row_stats row_lse(const float * logits) {
        float mx = logits[0];
        for (int i = 1; i < n_vocab; i++) mx = std::max(mx, logits[i]);
        double s = 0;
        for (int i = 0; i < n_vocab; i++) s += std::exp((double) (logits[i] - mx));
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

    bool run_jobs(std::vector<seq_job> & jobs, const std::vector<std::vector<option_spec>> & opts, int top_k, std::string & err) {
        const llama_pos P = (llama_pos) cached.size();
        auto cleanup = [&](size_t j0, size_t j1) { for (size_t j = j0; j < j1; j++) llama_memory_seq_rm(mem, seq0 + 1 + (llama_seq_id) (j - j0), -1, -1); };
        size_t j0 = 0;
        while (j0 < jobs.size()) {
            size_t j1 = j0, round_tokens = 0, L_max = 0;
            while (j1 < jobs.size() && (int) (j1 - j0) < n_qslots) {
                const size_t L = jobs[j1].tokens.size();
                const size_t new_max = std::max(L_max, L);
                const size_t cost = hybrid ? (j1 - j0 + 1) * new_max : round_tokens + L;
                if (j1 > j0 && (size_t) P + cost > (size_t) n_ctx) break;
                round_tokens += L; L_max = new_max; j1++;
            }
            if (j1 == j0) { err = "context too long for the KV budget"; return false; }
            for (size_t j = j0; j < j1; j++) llama_memory_seq_cp(mem, seq0, seq0 + 1 + (llama_seq_id) (j - j0), -1, -1);

            struct entry { int job; int ti; };
            std::vector<entry> ents;
            for (size_t j = j0; j < j1; j++) {
                size_t L = hybrid ? L_max : jobs[j].tokens.size();
                for (size_t t = 0; t < L; t++) ents.push_back({ (int) j, (int) t });
            }
            for (size_t e0 = 0; e0 < ents.size();) {
                size_t e1 = std::min(ents.size(), e0 + (size_t) n_batch);
                common_batch_clear(batch);
                std::vector<std::pair<int,int>> want;
                size_t e = e0;
                for (; e < e1; e++) {
                    auto & jb = jobs[ents[e].job];
                    int ti = ents[e].ti;
                    const bool is_pad = ti >= (int) jb.tokens.size();
                    auto it = is_pad ? jb.logit_pos.end() : std::find(jb.logit_pos.begin(), jb.logit_pos.end(), ti);
                    bool want_logits = it != jb.logit_pos.end();
                    // a chunk may not read more logits rows than the context's output buffer holds
                    if (want_logits && max_outputs > 0 && (int) want.size() >= max_outputs && batch.n_tokens > 0) break;
                    if (want_logits) want.push_back({ batch.n_tokens, (int) (it - jb.logit_pos.begin()) });
                    common_batch_add(batch, is_pad ? pad_token : jb.tokens[ti], P + ti, { seq0 + 1 + (llama_seq_id) (ents[e].job - (int) j0) }, want_logits);
                }
                e1 = e;
                int rc = llama_decode(ctx, batch);
                if (rc != 0) { err = "llama_decode failed on questions (rc=" + std::to_string(rc) + ")"; cleanup(j0, j1); return false; }
                for (auto & w : want) {
                    auto & job = jobs[ents[e0 + w.first].job];
                    const float * logits = llama_get_logits_ith(ctx, w.first);
                    if (!logits) { err = "null logits"; cleanup(j0, j1); return false; }
                    row_stats st = row_lse(logits);
                    auto lp = [&](llama_token t) { return (double) (logits[t] - st.mx) - st.lse; };
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
                        if (top_k > 0) job.top = topk_json(logits, st, top_k);
                    } else {
                        const auto & spec = opts[job.q_idx][job.opt_idx];
                        if (job.tok_logprobs.empty()) job.tok_logprobs.assign(spec.seq_tokens.size(), 0.0);
                        job.tok_logprobs[w.second] = lp(spec.seq_tokens[w.second]);
                    }
                }
                e0 = e1;
            }
            cleanup(j0, j1);
            j0 = j1;
        }
        return true;
    }

    // ------------------------------------------------------------------ request
    // {"system", "messages":[{role,content}...] (optional prior turns), "context", "separator",
    //  "enable_thinking", "assistant_prefix", "option_prefix", "scoring", "top_k", "questions":[{id,text,options[,scoring]}]}
    json judge(const json & req, const slot_cache_fn & slot_caches) {
        const double t0 = now_ms();
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

        std::vector<llama_tokens> full(qs.size());
        std::vector<std::vector<option_spec>> opts(qs.size());
        std::vector<std::string> scoring(qs.size());
        for (size_t i = 0; i < qs.size(); i++) {
            const auto & q = qs[i];
            std::string text = render(system, messages, context + separator + q.value("text", ""), enable_thinking) + assistant_prefix;
            full[i] = tokenize_prompt(text);
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
        llama_tokens prefix(full[0].begin(), full[0].begin() + P);

        std::string err, source;
        const double t1 = now_ms();
        int n_new = ensure_prefix(prefix, slot_caches ? slot_caches() : std::vector<slot_cache>{}, source, err);
        if (n_new < 0) throw std::runtime_error(err);
        const double t2 = now_ms();

        std::vector<seq_job> jobs; size_t n_qtok = 0;
        for (size_t i = 0; i < qs.size(); i++) {
            llama_tokens suffix(full[i].begin() + P, full[i].end());
            if (scoring[i] == "sequence") {
                for (size_t o = 0; o < opts[i].size(); o++) {
                    seq_job jb; jb.q_idx = (int) i; jb.opt_idx = (int) o; jb.tokens = suffix;
                    const auto & st = opts[i][o].seq_tokens;
                    for (size_t k = 0; k < st.size(); k++) jb.logit_pos.push_back((int) (suffix.size() - 1 + k));
                    jb.tokens.insert(jb.tokens.end(), st.begin(), st.end());
                    jb.tokens.pop_back();
                    n_qtok += jb.tokens.size(); jobs.push_back(std::move(jb));
                }
            } else {
                seq_job jb; jb.q_idx = (int) i; jb.opt_idx = -1; jb.tokens = suffix;
                jb.logit_pos.push_back((int) suffix.size() - 1);
                n_qtok += jb.tokens.size(); jobs.push_back(std::move(jb));
            }
        }
        if (!run_jobs(jobs, opts, top_k, err)) throw std::runtime_error(err);
        const double t3 = now_ms();

        json out;
        out["model"] = alias;
        out["n_prefix_tokens"] = P;
        out["n_prefix_computed"] = n_new;
        out["prefix_source"] = source;
        out["n_question_tokens"] = n_qtok;
        out["n_sequences"] = jobs.size();
        out["timings"] = { {"tokenize_ms", t1 - t0}, {"prefill_ms", t2 - t1}, {"questions_ms", t3 - t2}, {"total_ms", now_ms() - t0} };
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
