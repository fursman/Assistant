// Voice Assistant indicator — the assistant's phase in the GNOME top bar.
//
// The GNOME counterpart of the waybar module: it reads the same status file
// the assistant writes on every phase change (~/.local/state/voice-assistant/
// status, JSON with `class: [state, backend]`) and draws it as a robot head.
// A robot rather than a microphone because GNOME already puts a microphone
// in the same corner whenever any app is recording, and two identical icons
// side by side say nothing.
//
//   down       robot asleep, dark grey        (service not running)
//   off        robot, dark grey               (voice mode is off)
//   ready      robot, white                   (listening for speech)
//   listening  robot, red    + "listening"    (recording your turn)
//   thinking   robot, blue   + "thinking"     (model / tools working)
//   speaking   robot, green  + "speaking"     (playing the reply)
//
// Left click opens the conversation: a header saying what is happening and
// who is answering, three buttons (copy, new conversation, on/off), and the
// transcript filling the rest of the screen. Right click is the quick menu
// with the full set of controls. Both are the same
// PopupMenu with the transcript shown or hidden, because a panel button has
// one menu.
//
// Left click used to toggle voice mode. It no longer does: the transcript is
// what you reach for most, and muting has a key binding (SUPER+ALT on GNOME)
// plus its own item in both menus.
//
// The state directory is watched with a file monitor, so updates land within
// milliseconds of the assistant writing them; a slow poll reconciles anything
// a missed event or a dead process would otherwise leave stale.

import GObject from 'gi://GObject';
import St from 'gi://St';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Clutter from 'gi://Clutter';
import Pango from 'gi://Pango';

import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

const HOME = GLib.get_home_dir();
const STATE_DIR = GLib.build_filenamev([HOME, '.local', 'state', 'voice-assistant']);
const STATUS_FILE = GLib.build_filenamev([STATE_DIR, 'status']);
const PID_FILE = GLib.build_filenamev([STATE_DIR, 'voice-assistant.pid']);
const TRANSCRIPT_FILE = GLib.build_filenamev([STATE_DIR, 'transcript.json']);
// Turns rendered in the menu. The file keeps more; this is what fits.
const TRANSCRIPT_SHOWN = 12;

// Per-role colour and face, as Pango span attributes. Markup rather than one
// style class per label, because consecutive entries are rendered into ONE
// text actor: a mouse selection lives inside a single ClutterText, so separate
// labels could never be selected together, and this is what lets a selection
// run across entries and across changes of font.
const ROLE_MARKUP = {
    you:       'foreground="#99c1f1" weight="bold"',
    assistant: 'foreground="#ffffff"',
    tool:      'foreground="#9a9996" font_family="monospace" size="smaller"',
    thinking:  'foreground="#9a9996" style="italic" size="smaller"',
    system:    'foreground="#f9f06b" style="italic" size="smaller"',
};
const ASSISTANT_BIN = GLib.build_filenamev([HOME, '.local', 'bin', 'assistant']);
const POLL_SECONDS = 5;

const STATES = {
    off:       {icon: 'robot',        label: '',          title: 'Voice mode off'},
    ready:     {icon: 'robot',        label: '',          title: 'Ready — say something'},
    listening: {icon: 'robot',        label: 'listening', title: 'Listening'},
    thinking:  {icon: 'robot',        label: 'thinking',  title: 'Thinking'},
    speaking:  {icon: 'robot',        label: 'speaking',  title: 'Speaking'},
    down:      {icon: 'robot-asleep', label: '',          title: 'Assistant not running'},
};
const STATE_CLASSES = Object.keys(STATES).map(s => `voice-${s}`);

const BACKEND_LABEL = {
    claude: 'Claude',
    local: 'Local Qwen3.8',
    dsh: 'DeepSeek Harness · Qwen3.8',
};

function readFile(path) {
    try {
        const [ok, bytes] = GLib.file_get_contents(path);
        return ok ? new TextDecoder().decode(bytes).trim() : null;
    } catch (e) {
        return null;
    }
}

function assistantPid() {
    const text = readFile(PID_FILE);
    if (!text || !/^\d+$/.test(text))
        return null;
    return GLib.file_test(`/proc/${text}`, GLib.FileTest.IS_DIR) ? text : null;
}

function readStatus() {
    // Returns {state, backend}. `down` when the assistant is not running,
    // whatever the file says: it writes `off` on a clean exit but a crash
    // or a reboot leaves whatever was last written.
    const pid = assistantPid();
    if (!pid)
        return {state: 'down', backend: null, pid: null};
    let state = 'off', backend = null;
    const text = readFile(STATUS_FILE);
    if (text) {
        try {
            const data = JSON.parse(text);
            if (Array.isArray(data.class)) {
                if (data.class[0] in STATES)
                    state = data.class[0];
                backend = data.class[1] ?? null;
            }
        } catch (e) {
            // A partial write is impossible (the assistant renames into place),
            // so this is an old or foreign file; treat it as unknown state.
        }
    }
    let model = null, effort = null, models = [], efforts = [];
    if (text) {
        try {
            const d = JSON.parse(text);
            model = d.model ?? null;
            effort = d.effort ?? null;
            models = Array.isArray(d.models) ? d.models : [];
            efforts = Array.isArray(d.efforts) ? d.efforts : [];
        } catch (e) {
            // handled above
        }
    }
    return {state, backend, pid, model, effort, models, efforts};
}

function readTranscript() {
    // The conversation lives here rather than in notifications: GNOME queues
    // banners, ignores the expiry an app asks for, and clearing one early also
    // deletes it from the message list. A surface we own has none of that.
    const text = readFile(TRANSCRIPT_FILE);
    if (!text)
        return [];
    try {
        const data = JSON.parse(text);
        return Array.isArray(data.turns) ? data.turns : [];
    } catch (e) {
        return [];   // written atomically, so this is an old or foreign file
    }
}

function spawn(argv) {
    // Output is captured rather than silenced. Silencing it hid a real bug for
    // a while: the installed `assistant` was older than the extension and
    // rejected --model, so clicking a menu item printed a usage error into the
    // void and looked like nothing happening at all.
    try {
        const proc = Gio.Subprocess.new(argv,
            Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_PIPE);
        proc.communicate_utf8_async(null, null, (p, res) => {
            try {
                const [, out, err] = p.communicate_utf8_finish(res);
                if (!p.get_successful()) {
                    const why = (err || out || '').trim().split('\n')[0];
                    logError(new Error(`${argv.join(' ')}: ${why}`),
                        'voice-assistant-indicator');
                }
            } catch (e) {
                logError(e, 'voice-assistant-indicator');
            }
        });
    } catch (e) {
        logError(e, `voice-assistant-indicator: ${argv.join(' ')} failed`);
    }
}

const VoiceIndicator = GObject.registerClass(
class VoiceIndicator extends PanelMenu.Button {
    _init(iconDir) {
        super._init(0.5, 'voice-assistant', false);

        this._icons = {};
        for (const name of ['robot', 'robot-asleep']) {
            // The -symbolic suffix is what makes St recolour it from CSS.
            this._icons[name] = Gio.icon_new_for_string(
                GLib.build_filenamev([iconDir, `${name}-symbolic.svg`]));
        }

        this._box = new St.BoxLayout({
            style_class: 'voice-indicator',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._icon = new St.Icon({
            style_class: 'voice-icon',
            gicon: this._icons.robot,
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._label = new St.Label({
            style_class: 'voice-label',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._box.add_child(this._icon);
        this._box.add_child(this._label);
        this.add_child(this._box);

        // Header: what is happening and who is answering, then three buttons.
        // In the transcript view these are the only controls, so the transcript
        // gets the rest of the screen; the full set lives on right click.
        this._header = new PopupMenu.PopupBaseMenuItem({reactive: false, can_focus: false});
        const row = new St.BoxLayout({style_class: 'voice-header', x_expand: true});
        this._headerLabel = new St.Label({
            style_class: 'voice-header-label',
            x_expand: true,
            y_align: Clutter.ActorAlign.CENTER,
        });
        row.add_child(this._headerLabel);
        const button = (icon, name, onClick) => {
            const b = new St.Button({style_class: 'voice-header-button', can_focus: true});
            b.set_child(new St.Icon({icon_name: icon, style_class: 'voice-header-icon'}));
            b.accessible_name = name;
            b.connect('clicked', onClick);
            row.add_child(b);
            return b;
        };
        button('edit-copy-symbolic', 'Copy transcript', () => this._copyTranscript());
        button('document-new-symbolic', 'New conversation', () => {
            this.menu.close();
            this._signal('USR2');
        });
        this._powerButton = button('system-shutdown-symbolic', 'Voice mode on or off', () => {
            this.menu.close();
            this._signal('USR1');
        });
        this._header.add_child(row);
        this.menu.addMenuItem(this._header);
        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        // The transcript, oldest at the top so it reads downwards.
        this._transcriptBox = new St.BoxLayout({
            vertical: true,
            style_class: 'voice-transcript-box',
        });
        this._transcriptScroll = new St.ScrollView({
            style_class: 'voice-transcript',
            hscrollbar_policy: St.PolicyType.NEVER,
            vscrollbar_policy: St.PolicyType.AUTOMATIC,
        });
        // St.ScrollView took a single child from GNOME 46; add_actor before it.
        if (this._transcriptScroll.set_child)
            this._transcriptScroll.set_child(this._transcriptBox);
        else
            this._transcriptScroll.add_actor(this._transcriptBox);
        const transcriptItem = new PopupMenu.PopupBaseMenuItem({
            reactive: false,
            can_focus: false,
        });
        transcriptItem.add_child(this._transcriptScroll);
        this.menu.addMenuItem(transcriptItem);
        this._transcriptItem = transcriptItem;
        this._transcriptSig = null;
        this._lastTurns = [];
        this._runs = [];
        this._showingTranscript = true;

        this._toggleItem = new PopupMenu.PopupMenuItem('Turn voice mode on');
        this._toggleItem.connect('activate', () => this._signal('USR1'));
        this.menu.addMenuItem(this._toggleItem);

        const fresh = new PopupMenu.PopupMenuItem('New conversation');
        fresh.connect('activate', () => this._signal('USR2'));
        this.menu.addMenuItem(fresh);

        const swap = new PopupMenu.PopupMenuItem('Swap backend');
        swap.connect('activate', () => spawn([ASSISTANT_BIN, '--swap']));
        this.menu.addMenuItem(swap);

        // Claude's model and effort. Only meaningful on the Claude backend, so
        // both are hidden otherwise rather than offering a setting that would
        // sit unused until you switched back.
        this._modelMenu = new PopupMenu.PopupSubMenuMenuItem('Model');
        this.menu.addMenuItem(this._modelMenu);
        this._effortMenu = new PopupMenu.PopupSubMenuMenuItem('Effort');
        this.menu.addMenuItem(this._effortMenu);
        this._choiceSig = null;

        const startSep = new PopupMenu.PopupSeparatorMenuItem();
        this.menu.addMenuItem(startSep);
        const start = new PopupMenu.PopupMenuItem('Start the assistant service');
        start.connect('activate', () =>
            spawn(['systemctl', '--user', 'start', 'voice-assistant.service']));
        this.menu.addMenuItem(start);
        this._startItem = start;
        // Everything that is hidden in the transcript view. The model and
        // effort submenus are handled in _updateChoices, which also gates them.
        this._controls = [this._toggleItem, fresh, swap, startSep];

        this._current = null;
        this._installClickHandling();

        // An open PopupMenu takes a keyboard grab, so the desktop shortcut for
        // muting never reaches GNOME while the transcript is up. The menu's own
        // actor does see the keys, so catch the chord here and act on it. It is
        // a modifier-only binding (<Super>Alt_L and <Alt>Super_L), which
        // arrives as whichever of the two was pressed second.
        this.menu.actor.connect('key-press-event', (_a, event) => {
            const sym = event.get_key_symbol();
            const mods = event.get_state();
            // Ctrl+C on a mouse selection. The menu holds the grab, so this is
            // the only place the key can be seen.
            if ((sym === Clutter.KEY_c || sym === Clutter.KEY_C) &&
                (mods & Clutter.ModifierType.CONTROL_MASK)) {
                for (const ct of this._runs ?? []) {
                    const sel = ct.get_selection();
                    if (sel) {
                        St.Clipboard.get_default().set_text(St.ClipboardType.CLIPBOARD, sel);
                        return Clutter.EVENT_STOP;
                    }
                }
            }
            // Super arrives as MOD4_MASK in practice; SUPER_MASK is often
            // simply unset, which is why only the Alt-first order worked.
            const superHeld = (mods & (Clutter.ModifierType.SUPER_MASK |
                                       Clutter.ModifierType.MOD4_MASK)) !== 0;
            const altHeld = (mods & Clutter.ModifierType.MOD1_MASK) !== 0;
            const isAlt = sym === Clutter.KEY_Alt_L || sym === Clutter.KEY_Alt_R;
            const isSuper = sym === Clutter.KEY_Super_L || sym === Clutter.KEY_Super_R;
            if ((isAlt && superHeld) || (isSuper && altHeld)) {
                this.menu.close();
                this._signal('USR1');
                return Clutter.EVENT_STOP;
            }
            return Clutter.EVENT_PROPAGATE;
        });
    }

    // Left click toggles voice mode, like clicking the waybar module; any
    // other button opens the menu as a panel button normally does.
    //
    // GNOME 49+ panel buttons open their menu from a ClutterClickGesture
    // (`_clickGesture`), which recognises on press and does not look at the
    // button, so that gesture is switched off and replaced with one that
    // does. Older shells route clicks through vfunc_event instead, handled
    // below.
    _installClickHandling() {
        if (!this._clickGesture)
            return;
        this._clickGesture.set_enabled(false);
        const gesture = new Clutter.ClickGesture();
        gesture.set_recognize_on_press(true);
        gesture.connect('recognize', () => this._onClick(Clutter.get_current_event()));
        this.add_action(gesture);
        this._ownGesture = gesture;
    }

    vfunc_event(event) {
        if (this._ownGesture)
            return Clutter.EVENT_PROPAGATE;
        if (event.type() === Clutter.EventType.BUTTON_PRESS) {
            this._onClick(event);
            return Clutter.EVENT_STOP;
        }
        return super.vfunc_event(event);
    }

    _onClick(event) {
        let button = 1;
        try {
            button = event?.get_button?.() ?? 1;
        } catch (e) {
            button = 1;
        }
        // One menu, two shapes. Left is the conversation under the header;
        // right is the quick menu with the full set of controls.
        const wantTranscript = button === 1;
        if (this.menu?.isOpen && this._showingTranscript === wantTranscript) {
            this.menu.close();
            return;
        }
        this._showingTranscript = wantTranscript;
        this._applyMode();
        this.menu?.open();
    }

    _applyMode() {
        const t = this._showingTranscript;
        this._transcriptItem.visible = t;
        for (const c of this._controls)
            c.visible = !t;
        const status = this._current;
        if (status) {
            this._toggleItem.label.text =
                status.state === 'off' ? 'Turn voice mode on' : 'Turn voice mode off';
            this._toggleItem.sensitive = status.state !== 'down';
            this._startItem.visible = !t && status.state === 'down';
            this._updateChoices(status);
        }
        if (t)
            this._sizeTranscript();
    }

    _signal(sig) {
        const pid = assistantPid();
        if (pid)
            spawn(['kill', `-${sig}`, pid]);
    }

    update(status) {
        this._current = status;
        const spec = STATES[status.state];
        this._icon.gicon = this._icons[spec.icon];
        this._label.text = spec.label;
        this._label.visible = spec.label !== '';

        for (const cls of STATE_CLASSES)
            this._box.remove_style_class_name(cls);
        this._box.add_style_class_name(`voice-${status.state}`);

        // "Listening · Claude · fable · xhigh": everything about who is
        // answering, in one line, so none of it needs a menu to find out.
        const parts = [spec.title];
        if (status.backend)
            parts.push(BACKEND_LABEL[status.backend] ?? status.backend);
        if (status.backend === 'claude') {
            if (status.model) parts.push(status.model);
            if (status.effort) parts.push(status.effort);
        }
        this._headerLabel.text = parts.join(' · ');
        this._powerButton.opacity = status.state === 'off' ? 128 : 255;
        this._applyMode();
    }

    _copyTranscript() {
        const names = {you: 'You', assistant: 'Assistant', tool: 'Tool',
            system: 'System', thinking: 'Thinking'};
        const text = this._lastTurns
            .map(t => `${names[t.role] ?? t.role}: ${t.text}`)
            .join('\n\n');
        St.Clipboard.get_default().set_text(St.ClipboardType.CLIPBOARD, text);
        Main.notify('Copied transcript', `${this._lastTurns.length} entries`);
        this.menu?.close();
    }

    // Rebuilt only when the list of options changes; otherwise just the tick
    // moves, so opening a submenu does not fight a rebuild underneath it.
    _updateChoices(status) {
        const onClaude = status.backend === 'claude' && !this._showingTranscript;
        this._modelMenu.visible = onClaude && status.models.length > 0;
        this._effortMenu.visible = onClaude && status.efforts.length > 0;
        if (!onClaude)
            return;

        const sig = `${status.models.join(',')}|${status.efforts.join(',')}`;
        if (sig !== this._choiceSig) {
            this._choiceSig = sig;
            this._modelItems = this._fillChoices(this._modelMenu, status.models, '--model');
            this._effortItems = this._fillChoices(this._effortMenu, status.efforts, '--effort');
        }
        this._modelMenu.label.text = `Model: ${status.model ?? '?'}`;
        this._effortMenu.label.text = `Effort: ${status.effort ?? '?'}`;
        for (const [name, item] of this._modelItems ?? [])
            item.setOrnament(name === status.model
                ? PopupMenu.Ornament.DOT : PopupMenu.Ornament.NONE);
        for (const [name, item] of this._effortItems ?? [])
            item.setOrnament(name === status.effort
                ? PopupMenu.Ornament.DOT : PopupMenu.Ornament.NONE);
    }

    _fillChoices(submenu, names, flag) {
        submenu.menu.removeAll();
        const items = [];
        for (const name of names) {
            const item = new PopupMenu.PopupMenuItem(name);
            item.connect('activate', () => spawn([ASSISTANT_BIN, flag, name]));
            submenu.menu.addMenuItem(item);
            items.push([name, item]);
        }
        return items;
    }

    // Sized against the monitor rather than a fixed number of pixels, so it
    // fills the screen on this display and still fits on a smaller one. The
    // reserve covers the panel plus the controls below the transcript.
    _sizeTranscript() {
        const mon = Main.layoutManager?.primaryMonitor;
        if (!mon)
            return;
        // Only the header sits above the transcript now, and nothing below.
        const reserve = (Main.panel?.height ?? 32) + 90;
        const h = Math.max(200, mon.height - reserve);
        this._transcriptScroll.style = `max-height: ${h}px;`;
    }

    // Wrapping has to be configured AFTER the label is parented: St.Label
    // rebuilds its ClutterText when the stylesheet applies, which puts
    // ellipsize back. Width comes from the stylesheet and is what the text
    // wraps against; without all three the line either runs off the side of
    // the screen or is cut with an ellipsis.
    _wrap(label) {
        const ct = label.clutter_text;
        ct.single_line_mode = false;
        ct.ellipsize = Pango.EllipsizeMode.NONE;
        ct.line_wrap = true;
        ct.line_wrap_mode = Pango.WrapMode.WORD_CHAR;
    }

    _markup(role, text) {
        const attrs = ROLE_MARKUP[role] ?? ROLE_MARKUP.assistant;
        return `<span ${attrs}>${GLib.markup_escape_text(text, -1)}</span>`;
    }

    // One selectable text actor holding a run of entries.
    _addRun(markup) {
        const label = new St.Label({style_class: 'voice-transcript-run'});
        label.x_expand = true;
        label.reactive = true;      // or the pointer never reaches the text
        label.can_focus = true;
        this._transcriptBox.add_child(label);
        this._wrap(label);
        const ct = label.clutter_text;
        ct.set_markup(markup);
        ct.selectable = true;
        ct.editable = false;
        ct.cursor_visible = false;
        this._runs.push(ct);
    }

    _addCode(code) {
        if (!code)
            return;
        const button = new St.Button({
            style_class: 'voice-turn-code',
            x_expand: true,
            can_focus: true,
        });
        const label = new St.Label({text: code});
        button.set_child(label);
        this._transcriptBox.add_child(button);
        this._wrap(label);
        // St.Clipboard, not an external tool: the Shell owns the selection and
        // outlives any process, which is exactly what a clipboard needs.
        button.connect('clicked', () => {
            St.Clipboard.get_default().set_text(St.ClipboardType.CLIPBOARD, code);
            Main.notify('Copied to clipboard', code.split('\n')[0]);
            this.menu?.close();
        });
    }

    updateTranscript(turns) {
        this._lastTurns = turns;
        const shown = turns.slice(-TRANSCRIPT_SHOWN);
        // Rebuilding on every poll would fight the user's scrolling and drop
        // their selection, so only touch it when the content actually changed.
        const sig = shown.map(t => `${t.role}\u0000${t.at}\u0000${t.text.length}`).join('|');
        if (sig === this._transcriptSig)
            return;
        this._transcriptSig = sig;
        this._transcriptBox.destroy_all_children();
        this._runs = [];

        if (!shown.length) {
            this._transcriptBox.add_child(new St.Label({
                style_class: 'voice-turn-empty',
                text: 'Nothing said yet',
            }));
            return;
        }

        // Entries accumulate into one run; a code block flushes the run and
        // sits between runs as its own clickable row, so a selection crosses
        // everything except a code block.
        let pending = [];
        const flush = () => {
            if (pending.length)
                this._addRun(pending.join('\n\n'));
            pending = [];
        };
        for (const turn of shown) {
            if (turn.role === 'assistant' && turn.text.includes('```')) {
                let last = 0;
                CODE_FENCE.lastIndex = 0;
                let m;
                while ((m = CODE_FENCE.exec(turn.text)) !== null) {
                    const before = turn.text.slice(last, m.index).trim();
                    if (before)
                        pending.push(this._markup('assistant', before));
                    flush();
                    this._addCode(m[1].trim());
                    last = m.index + m[0].length;
                }
                const after = turn.text.slice(last).trim();
                if (after)
                    pending.push(this._markup('assistant', after));
            } else {
                pending.push(this._markup(turn.role, turn.text));
            }
        }
        flush();

        // Scroll to the newest, once the labels have been laid out.
        GLib.idle_add(GLib.PRIORITY_DEFAULT_IDLE, () => {
            const adj = this._transcriptScroll.vadjustment
                ?? this._transcriptScroll.vscroll?.adjustment;
            if (adj)
                adj.value = Math.max(0, adj.upper - adj.page_size);
            return GLib.SOURCE_REMOVE;
        });
    }
});

export default class VoiceAssistantIndicatorExtension extends Extension {
    enable() {
        this._indicator = new VoiceIndicator(GLib.build_filenamev([this.path, 'icons']));
        Main.panel.addToStatusArea(this.uuid, this._indicator, 0, 'right');

        // The assistant creates the directory itself; creating it here too
        // means the monitor exists before the first run ever happens.
        GLib.mkdir_with_parents(STATE_DIR, 0o755);
        try {
            this._monitor = Gio.File.new_for_path(STATE_DIR)
                .monitor_directory(Gio.FileMonitorFlags.NONE, null);
            this._monitorId = this._monitor.connect('changed', () => this._sync());
        } catch (e) {
            logError(e, 'voice-assistant-indicator: file monitor failed, polling only');
        }

        this._timerId = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, POLL_SECONDS, () => {
            this._sync();
            return GLib.SOURCE_CONTINUE;
        });

        this._sync();
    }

    _sync() {
        if (!this._indicator)
            return;
        this._indicator.update(readStatus());
        this._indicator.updateTranscript(readTranscript());
    }

    disable() {
        if (this._timerId) {
            GLib.Source.remove(this._timerId);
            this._timerId = null;
        }
        if (this._monitor) {
            if (this._monitorId)
                this._monitor.disconnect(this._monitorId);
            this._monitor.cancel();
            this._monitor = null;
            this._monitorId = null;
        }
        this._indicator?.destroy();
        this._indicator = null;
    }
}
