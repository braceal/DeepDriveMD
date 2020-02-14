import os
import click


def validate_path(ctx, param, value):
    """Add abspath to non-None file."""
    if value:
        path = os.path.abspath(value)
        if not os.path.exists(path):
            raise click.BadParameter(f'path does not exist {path}')
        return path

def validate_positive(ctx, param, value):
    """Check that param value is greater than 0."""
    if value < 0:
        raise click.BadParameter(f'must be greater than or equal to 0, currently {value}')
    return value

def validate_between_zero_and_one(ctx, param, value):
    """Check that param value is between 0 and 1"""
    if value < 0 or value > 1:
        raise click.BadParameter('must be greater than or equal to 0 and'
                                 f'less than or equal to 1, currently {value}')
    return value
