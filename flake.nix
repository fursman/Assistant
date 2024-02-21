{
  description = "A NixOS Flake for NixOS Assistant LLM AI for Wayland";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Import nixpkgs without any overlays applied
        pkgs = import nixpkgs {
          inherit system;
        };

        # Define your Python environment using the customized package set
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.keyring
          ps.notify2
          ps.openai
        ]);
      in {
        packages.assistant = pkgs.stdenv.mkDerivation {
          name = "assistant";
          buildInputs = [ pythonEnv pkgs.ffmpeg-full pkgs.portaudio pkgs.gnome.zenity ];
          # Additional configuration...
        };

        # Additional configuration...
      }
    );
}
