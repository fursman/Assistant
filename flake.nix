{
  description = "A NixOS Flake for NixOS Assistant LLM AI for Wayland";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Overlay to customize Python packages
        customPythonOverlay = final: prev: {
          python3Packages = prev.python3Packages // {
            openai = prev.python3Packages.openai.overrideAttrs (oa: {
              doCheck = false;  # Disable build tests for openai
            });
          };
        };

        # Import nixpkgs with the overlay applied
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ customPythonOverlay ];
        };

        # Define your Python environment using the customized package set
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.keyring
          ps.notify2
          ps.openai  # Uses the version with tests disabled
        ]);
      in {
        packages.assistant = pkgs.stdenv.mkDerivation {
          name = "assistant";
          buildInputs = [ pythonEnv pkgs.ffmpeg pkgs.portaudio ];
          # Additional configuration...
        };

        # Additional configuration...
      }
    );
}
