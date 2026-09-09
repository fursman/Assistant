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
// Left click toggles voice mode (SIGUSR1 to the pid file, the same thing the
// key binding does). Right click opens a menu with the state, a new
// conversation, and the model swap.
//
// The state directory is watched with a file monitor, so updates land within
// milliseconds of the assistant writing them; a slow poll reconciles anything
// a missed event or a dead process would otherwise leave stale.

import GObject from 'gi://GObject';
import St from 'gi://St';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Clutter from 'gi://Clutter';

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
    return {state, backend, pid};
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
    try {
        Gio.Subprocess.new(argv,
            Gio.SubprocessFlags.STDOUT_SILENCE | Gio.SubprocessFlags.STDERR_SILENCE);
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

        this._header = new PopupMenu.PopupMenuItem('Voice Assistant', {reactive: false});
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
        this._transcriptSig = null;
        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._toggleItem = new PopupMenu.PopupMenuItem('Turn voice mode on');
        this._toggleItem.connect('activate', () => this._signal('USR1'));
        this.menu.addMenuItem(this._toggleItem);

        const fresh = new PopupMenu.PopupMenuItem('New conversation');
        fresh.connect('activate', () => this._signal('USR2'));
        this.menu.addMenuItem(fresh);

        const swap = new PopupMenu.PopupMenuItem('Swap model');
        swap.connect('activate', () => spawn([ASSISTANT_BIN, '--swap']));
        this.menu.addMenuItem(swap);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        const start = new PopupMenu.PopupMenuItem('Start the assistant service');
        start.connect('activate', () =>
            spawn(['systemctl', '--user', 'start', 'voice-assistant.service']));
        this.menu.addMenuItem(start);
        this._startItem = start;

        this._current = null;
        this._installClickHandling();
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
        if (button === 1 && this._current?.state !== 'down')
            this._signal('USR1');
        else
            this.menu?.toggle();
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

        const backend = status.backend ? (BACKEND_LABEL[status.backend] ?? status.backend) : null;
        this._header.label.text = backend ? `${spec.title} · ${backend}` : spec.title;
        this._toggleItem.label.text =
            status.state === 'off' ? 'Turn voice mode on' : 'Turn voice mode off';
        this._toggleItem.sensitive = status.state !== 'down';
        this._startItem.visible = status.state === 'down';
    }

    updateTranscript(turns) {
        const shown = turns.slice(-TRANSCRIPT_SHOWN);
        // Rebuilding on every poll would fight the user's scrolling, so only
        // touch it when the content actually changed.
        const sig = shown.map(t => `${t.role}\u0000${t.at}\u0000${t.text.length}`).join('|');
        if (sig === this._transcriptSig)
            return;
        this._transcriptSig = sig;
        this._transcriptBox.destroy_all_children();

        if (!shown.length) {
            this._transcriptBox.add_child(new St.Label({
                style_class: 'voice-turn-empty',
                text: 'Nothing said yet',
            }));
            return;
        }

        for (const turn of shown) {
            const label = new St.Label({
                style_class: turn.role === 'you' ? 'voice-turn-you' : 'voice-turn-assistant',
                text: turn.text,
            });
            label.clutter_text.line_wrap = true;
            this._transcriptBox.add_child(label);
        }

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
