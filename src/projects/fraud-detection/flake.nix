{
  description = "Fraud Detection ML";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        pythonPackages = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPackages.pandas
            pythonPackages.numpy
            pythonPackages.scikit-learn
            pythonPackages.matplotlib
            pythonPackages.seaborn
          ];

          shellHook = ''
            echo "Fraud Detection Environment Loaded."
            echo "Python: $(python --version)"
          '';
        };
      }
    );
}

