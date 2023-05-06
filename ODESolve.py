# -*- coding: utf-8 -*-
"""
Created on Fri May  5 18:45:41 2023

@author: fink4
"""

def euler(func, t, dt, y):
  return dt * func(t, y)


def rk4(func, t, dt, y):
  _one_sixth = 1/6
  half_dt = dt * 0.5

  k1 = func(t, y)
  k2 = func(t + half_dt, y + half_dt * k1)
  k3 = func(t + half_dt, y + half_dt * k2)
  k4 = func(t + dt, y + dt * k3)
  
  return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth 
