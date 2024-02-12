{
  description = "A flake for installing NixOS Assistant with openai package";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Import nixpkgs with potential overlays or custom configurations
        pkgs = import nixpkgs {
          inherit system;
        };

        # Custom Python environment with openai, skipping tests
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          # Other packages as needed
          (ps.openai.overrideAttrs (oldAttrs: {
            doCheck = false; # Skip tests
          }))
        ]);
      in {
        packages.assistant = pkgs.stdenv.mkDerivation {
          name = "assistant";
          buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio ];
          dontUnpack = true;
          installPhase = ''
            mkdir -p $out/bin
            cp ${self}/assistant.py $out/bin/assistant
            chmod +x $out/bin/assistant
            wrapProgram $out/bin/assistant \
              --prefix PATH : "${pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.portaudio pythonEnv ]}"
          '';
        };

        defaultPackage.${system} = self.packages.${system}.assistant;

        devShells.${system} = pkgs.mkShell {
          buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio ];
        };
      }
    );
}
