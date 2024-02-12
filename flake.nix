{
  description = "A flake for installing NixOS Assistant";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Define an overlay to specify the openai version
        openaiOverlay = final: prev: {
          python3Packages = prev.python3Packages // {
            openai = prev.python3Packages.openai.overrideAttrs (oldAttrs: rec {
              version = "1.6.1";
              src = final.fetchPypi {
                pname = "openai";
                version = "1.6.1";
                sha256 = "0d8sj9qfidq5jsfass3p1jx6iyjg6np9s7c5x2f7k32flbq2gv07"; # Update this with the correct hash
              };
            });
          };
        };

        # Import nixpkgs with the overlay applied
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ openaiOverlay ];
        };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.keyring
          ps.notify2
          ps.openai # This now refers to the overridden version
        ]);
      in {
        packages.assistant = pkgs.stdenv.mkDerivation {
          name = "assistant";
          src = self;
          buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio pkgs.makeWrapper ];
          dontUnpack = true;
          installPhase = ''
            mkdir -p $out/bin
            cp ${self}/assistant.py $out/bin/assistant
            chmod +x $out/bin/assistant
            wrapProgram $out/bin/assistant \
              --prefix PATH : ${lib.makeBinPath [ pkgs.ffmpeg pkgs.portaudio pythonEnv ]}
              # Include any other necessary runtime dependencies in the PATH
          '';
        };

        defaultPackage.${system} = self.packages.${system}.assistant;

        devShells.${system} = pkgs.mkShell {
          buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio ];
        };
      }
    );
}
