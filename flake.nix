{
  description = "A Flake for the Assistant application";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        lib = nixpkgs.lib;

        assistantSrc = self;

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.requests
          ps.numpy
          ps.notify2
          ps.keyring
          ps.websocket-client
          ps.pydub
          ps.openai

          ps.python-dotenv
          ps.sounddevice
          ps.pydub
        ]);

      in {
        packages.assistant = pkgs.stdenv.mkDerivation {
          name = "assistant";
          src = assistantSrc;

          buildInputs = [
            pythonEnv
            pkgs.ffmpeg-full
            pkgs.portaudio
            pkgs.wl-clipboard
            pkgs.zenity
          ];

          nativeBuildInputs = [ pkgs.makeWrapper ];

          installPhase = ''
            mkdir -p $out/bin $out/share/assistant/assets-audio $out/share/assistant/logs

            # Install Python script
            cp ${assistantSrc}/assistant.py $out/bin/assistant
            chmod +x $out/bin/assistant

            cp ${assistantSrc}/realtime-assistant.py $out/bin/realtime-assistant
            chmod +x $out/bin/realtime-assistant

            # Copy audio assets
            cp -r ${assistantSrc}/assets-audio/* $out/share/assistant/assets-audio/
          '';

          postFixup = ''
            wrapProgram $out/bin/assistant \
              --set AUDIO_ASSETS "$out/share/assistant/assets-audio" \
              --set LOG_DIR "/tmp/logs/assistant" \
              --prefix PATH : ${pkgs.ffmpeg-full}/bin
          '';
        };

        defaultPackage.${system} = self.packages.${system}.assistant;
      }
    );
}
