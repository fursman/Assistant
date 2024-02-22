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

        # Directly use the local source
        assistantSrc = self;

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

          nativeBuildInputs = [ pkgs.makeWrapper ];

          installPhase = ''
            mkdir -p $out/bin $out/share/assistant/assets-audio $out/share/assistant/logs

            # Install Python script
            cp ${assistantSrc}/assistant.py $out/bin/assistant
            chmod +x $out/bin/assistant

            # Copy audio assets
            cp -r ${assistantSrc}/assets-audio/* $out/share/assistant/assets-audio/
            cp -r ${assistantSrc}/logs/* $out/share/assistant/logs/
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
