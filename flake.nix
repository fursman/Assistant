{
  description = "A flake to install the NixOS AI LLM Assistant for Wayland";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
  flake-utils.lib.eachDefaultSystem (system:
    let
      # Import nixpkgs and extract both pkgs and lib
      pkgs = import nixpkgs {
        inherit system;
      };
      lib = pkgs.lib;
      
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        ps.pyaudio
        ps.numpy
        ps.keyring
        ps.notify2
        ps.openai
        # Add any other Python dependencies here
      ]);
    in {
      packages.assistant = pkgs.stdenv.mkDerivation {
        name = "assistant";
        src = self;
        buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio ]; # Ensure all external dependencies are included
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
