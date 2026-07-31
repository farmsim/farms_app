""" Run """

import os
import argparse

from farms_core import pylog
from farms_app.core.application import FARMSApplication


pylog.set_level("error")


def main():
    """ Main Application """
    parser = argparse.ArgumentParser(description="FARMS Application")
    parser.add_argument(
        "--options", "-o",
        default="options.yaml",
        help="Path to options YAML file (default: options.yaml in cwd)",
    )
    parser.add_argument("--title", "-t", default=None, help="Window title override")
    parser.add_argument("--enable", "-e", nargs="+", default=None, help="Extensions to enable on startup")
    parser.add_argument("--experiment", "-x", default=None, help="Path to experiment config file to load on startup")
    parser.add_argument("--log-level", "-l", default=None, choices=["debug", "info", "warning", "error"], help="Log level")
    args = parser.parse_args()

    if args.log_level:
        pylog.set_level(args.log_level)

    options = FARMSApplication.load_options(args.options)

    if args.title:
        options.title = args.title
    if args.enable:
        options.extension.auto_enable = args.enable

    # If experiment path is provided, ensure farmsim is enabled
    if args.experiment:
        if not options.extension.auto_enable:
            options.extension.auto_enable = []
        if "farmsim" not in options.extension.auto_enable:
            options.extension.auto_enable.append("farmsim")

    app = FARMSApplication(options, options_path=args.options)

    # If experiment path is provided, store it for FARMSIM extension to load
    if args.experiment:
        # Convert to absolute path to avoid issues with cwd changes in load_experiment
        app.experiment_path = os.path.abspath(args.experiment)

    app.run()


if __name__ == '__main__':
    main()
