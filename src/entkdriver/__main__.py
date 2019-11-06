"""A Python2 program for running the DeepDriveMD pipeline."""
import click
import Pyro4
from .deepdrive import EntkDriver


@click.command()
@click.option('--uri', help='URI for server DeepDriveMD object')
def main(uri):
    try:
        ddMD = Pyro4.Proxy(uri)
        driver = EntkDriver(ddMD)
        driver.run()
    except Exception as e:
        raise e


if __name__ == '__main__':
    main()
