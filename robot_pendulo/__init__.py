"""Simulador para entrenar con aprendizaje por refuerzo un robot de 4 ruedas
que equilibra un péndulo invertido sobre terrenos variados."""
from gymnasium.envs.registration import register

from .entorno import ConfigEntorno, RobotPenduloEnv
from .modelo import ParamsRobot
from .terreno import TIPOS_TERRENO

register(id="RobotPendulo-v0", entry_point="robot_pendulo.entorno:RobotPenduloEnv")

__all__ = ["RobotPenduloEnv", "ConfigEntorno", "ParamsRobot", "TIPOS_TERRENO"]
