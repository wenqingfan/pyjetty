#!/usr/bin/env python3

"""
    Class to store associated truth info to the detector level fj particles, 
    used primarily for applyinh pair efficeincy in the fast simulation.
    
  Author: Wenqing Fan (wenqing.fan@cern.ch)
"""

from __future__ import print_function

# Base class
from pyjetty.alice_analysis.process.base import common_base

################################################################
class EcorrInfo(common_base.CommonBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, **kwargs):
    super(EcorrInfo, self).__init__(**kwargs)

    # Store the associated truth info and particle charge
    self.particle_truth = None
    self.charge= 1000.
