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
          builtins.trace "Applying openai overlay" {}; # Trace to confirm overlay application
          python3Packages = prev.python3Packages // {
            openai = prev.python3Packages.openai.overrideAttrs (oldAttrs: rec {
              version = "1.6.1";
              src = final.fetchPypi {
                pname = "openai";
                version = "1.6.1";
                sha256 = "0000000000000000000000000000000000000000000000000000"; # Placeholder SHA256
              };
            });
          };
        };

        # Import nixpkgs with the overlay applied
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ openaiOverlay ];
        };

        # Use pythonEnv to include openai directly for testing
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.keyring
          ps.notify2
          (ps.openai.overrideAttrs (oldAttrs: rec { # Override directly here for testing
            version = "1.6.1";
            src = ps.fetchPypi {
              pname = "openai";
              version = "1.6.1";
              sha256 = "0000000000000000000000000000000000000000000000000000"; # Placeholder SHA256
            };
          }))
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
              --prefix PATH : ''${pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.portaudio pythonEnv ]}
          '';
        };

        defaultPackage.${system} = self.packages.${system}.assistant;

        devShells.${system} = pkgs.mkShell {
          buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio ];
        };
      }
    );
}
