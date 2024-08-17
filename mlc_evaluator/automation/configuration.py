import os


def config_from_env() -> dict:
    config = {}
    for the_var in ("CR_PAT", "CR_USER", "HF_TOKEN"):
        config[the_var] = os.getenv(the_var)
        if not config[the_var]:
            raise KeyError(f"Environment variable {the_var} needs to be set.")
