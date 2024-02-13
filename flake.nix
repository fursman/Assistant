{
  description = "A flake for the Assistant project with a specific openai version";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              python3Packages = prev.python3Packages // {
                openai = prev.python3Packages.buildPythonPackage rec {
                  pname = "openai";
                  version = "1.6.1";
                  src = prev.fetchPypi {
                    inherit pname version;
                    sha256 = "<SHA256_HASH>"; # Replace <SHA256_HASH> with the actual hash
                  };
                  doCheck = false; # Disable tests
                };
              };
            })
          ];
        };

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.keyring
          ps.notify2
          ps.openai # This now uses the specific version 1.6.1
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
              --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.portaudio pythonEnv ]}
          '';
        };

        defaultPackage.${system} = self.packages.${system}.assistant;
      }
    );
}
