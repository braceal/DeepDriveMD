import os
import click


def validate_path(ctx, param, value):
    """
    Adds abspath to non-None file
    """
    if value:
        path = os.path.abspath(value)
        if not os.path.exists(path):
            raise click.BadParameter(f'path does not exist {path}')
        return path

def validate_positive(ctx, param, value):
    """
    Checks that param value is greater than 0
    """
    if value < 0:
        raise click.BadParameter(f'must be greater than or equal to 0, currently {value}')
    return value
