{
  description = "A flake for the Assistant project with a specific openai version";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Overlay to override the openai package version
        overlay = self: super: {
          python3Packages = super.python3Packages // {
            openai = super.python3Packages.openai.overrideAttrs (oldAttrs: {
              src = super.fetchPypi {
                pname = "openai";
                version = "1.6.1";
                sha256 = "0000000000000000000000000000000000000000000000000000"; # Update this with the correct hash
              };
            });
          };
        };

        # Import nixpkgs with the overlay applied
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ overlay ];
        };

        # Define your Python environment
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.keyring
          ps.notify2
          ps.openai # This should now refer to the overridden version
        ]);
      in {
        # Define your NixOS configuration or package
        packages.assistant = pkgs.stdenv.mkDerivation {
          name = "assistant";
          src = self;
          buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio pkgs.makeWrapper ]; # Ensure all external dependencies are included
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
