from typing import Optional
from pydantic import BaseSettings
from dotenv import load_dotenv
from argparse import ArgumentParser


class Settings(BaseSettings):

    IPC: str
    RUNTIME: str
    HOST_IP: str
    HOST_PORT: int
    CONTAINER_PORT: int
    COMPOSE_PROJECT_NAME: str
    NVIDIA_VISIBLE_DEVICES: Optional[str]
    NVIDIA_DRIVER_CAPABILITIES: Optional[str]


def load_env():
    # Let the script caller define the .env file to use, e.g.:  python api.py --env .prod.env
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', default='.env',
                        help='Sets the environment file')

    args = parser.parse_args()
    load_dotenv(args.env)
