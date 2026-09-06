// Voice Assistant indicator — the assistant's phase in the GNOME top bar.
//
// The GNOME counterpart of the waybar module: it reads the same status file
// the assistant writes on every phase change (~/.local/state/voice-assistant/
// status, JSON with `class: [state, backend]`) and draws it as a pill, in the
// style of GNOME's own screen-recording indicator.
//
//   off        muted microphone, dimmed             (voice mode is off)
//   ready      microphone                           (listening for speech)
//   listening  red pill    "listening"              (recording your turn)
//   thinking   blue pill   "thinking"               (model / tools working)
//   speaking   green pill  "speaking"               (playing the reply)
//   down       crossed-out microphone, dimmed       (service not running)
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
const ASSISTANT_BIN = GLib.build_filenamev([HOME, '.local', 'bin', 'assistant']);
const POLL_SECONDS = 5;

const STATES = {
    off:       {icon: 'microphone-sensitivity-muted-symbolic', label: '',          title: 'Voice mode off'},
    ready:     {icon: 'audio-input-microphone-symbolic',       label: '',          title: 'Ready — say something'},
    listening: {icon: 'audio-input-microphone-symbolic',       label: 'listening', title: 'Listening'},
    thinking:  {icon: 'content-loading-symbolic',              label: 'thinking',  title: 'Thinking'},
    speaking:  {icon: 'audio-speakers-symbolic',               label: 'speaking',  title: 'Speaking'},
    down:      {icon: 'microphone-disabled-symbolic',          label: '',          title: 'Assistant not running'},
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
    _init() {
        super._init(0.5, 'voice-assistant', false);

        this._pill = new St.BoxLayout({
            style_class: 'voice-pill',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._icon = new St.Icon({
            style_class: 'voice-icon',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._label = new St.Label({
            style_class: 'voice-label',
            y_align: Clutter.ActorAlign.CENTER,
        });
        this._pill.add_child(this._icon);
        this._pill.add_child(this._label);
        this.add_child(this._pill);

        this._header = new PopupMenu.PopupMenuItem('Voice Assistant', {reactive: false});
        this.menu.addMenuItem(this._header);
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
    }

    // Left click toggles voice mode, like clicking the waybar module; anything
    // else (right click, touch) opens the menu as a panel button normally does.
    vfunc_event(event) {
        if (event.type() === Clutter.EventType.BUTTON_PRESS && event.get_button() === 1) {
            if (this._current?.state === 'down')
                this.menu.toggle();
            else
                this._signal('USR1');
            return Clutter.EVENT_STOP;
        }
        return super.vfunc_event(event);
    }

    _signal(sig) {
        const pid = assistantPid();
        if (pid)
            spawn(['kill', `-${sig}`, pid]);
    }

    update(status) {
        this._current = status;
        const spec = STATES[status.state];
        this._icon.icon_name = spec.icon;
        this._label.text = spec.label;
        this._label.visible = spec.label !== '';

        for (const cls of STATE_CLASSES)
            this._pill.remove_style_class_name(cls);
        this._pill.add_style_class_name(`voice-${status.state}`);

        const backend = status.backend ? (BACKEND_LABEL[status.backend] ?? status.backend) : null;
        this._header.label.text = backend ? `${spec.title} · ${backend}` : spec.title;
        this._toggleItem.label.text =
            status.state === 'off' ? 'Turn voice mode on' : 'Turn voice mode off';
        this._toggleItem.sensitive = status.state !== 'down';
        this._startItem.visible = status.state === 'down';
    }
});

export default class VoiceAssistantIndicatorExtension extends Extension {
    enable() {
        this._indicator = new VoiceIndicator();
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
