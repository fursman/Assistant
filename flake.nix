{
  description = "A description of your flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Define a custom package overlay
        customPythonPackages = pythonPackages: {
          openai = pythonPackages.openai.overrideAttrs (oldAttrs: {
            doCheck = false;  # This should disable the build tests for openai
          });
        };

        # Import nixpkgs with the overlay for Python packages
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (self: super: {
              python3Packages = super.python3Packages.overridePythonAttrs (oldAttrs: customPythonPackages super.python3Packages);
            })
          ];
        };

        # Define your Python environment with the custom package set
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          ps.pyaudio
          ps.numpy
          ps.keyring
          ps.notify2
          ps.openai  # This should now have tests disabled
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
