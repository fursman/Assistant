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

        assistantSrc = pkgs.fetchGit {
          url = "https://github.com/fursman/Assistant";
          ref = "main";
          # Note: fetchGit does not require sha256, but it has limitations outside of pure Nix environments.
        };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.notify2
          ps.keyring
        ]);

      in {
        packages.assistant = pkgs.stdenv.mkDerivation {
          name = "assistant";
          src = assistantSrc;

          buildInputs = [
            pythonEnv
            pkgs.ffmpeg-full
            pkgs.portaudio
            pkgs.gnome.zenity
            # Include the openai package if available in your nixpkgs version,
            # otherwise, you might need to use an overlay or package it yourself.
          ];

          installPhase = ''
            mkdir -p $out/bin $out/share/assistant/audio $out/var/log/assistant

            # Install Python script
            cp $src/assistant.py $out/bin/assistant
            chmod +x $out/bin/assistant

            # Copy audio assets
            cp -r $src/assets-audio/* $out/share/assistant/assets-audio/
            cp -r $src/logs/* $out/share/assistant/logs/
          '';

          postFixup = ''
            wrapProgram $out/bin/assistant \
              --set AUDIO_ASSETS "$out/share/assistant/assets-audio" \
              --set LOG_DIR "$out/share/assistant/logs"
          '';
        };

        defaultPackage.${system} = self.packages.${system}.assistant;
      }
    );
}
