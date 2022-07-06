#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of track information
  and do jet-finding, and save basic histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse
import sys

# Data analysis and plotting
import ROOT
import yaml
import numpy as np
import array 
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib

# Base class
from pyjetty.alice_analysis.process.user.substructure import process_data_base

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

def calculate_distance(p0, p1):
    
  dphiabs = math.fabs(p0.phi() - p1.phi())
  dphi = dphiabs

  if dphiabs > math.pi:
    dphi = 2*math.pi - dphiabs


  dy = p0.rap() - p1.rap()
  return math.sqrt(dy*dy + dphi*dphi)

class CorrelatorBuilder:
  def __init__(self, particle_list, jetpt):
    self.particle_list = particle_list
    self.mult = len(self.particle_list)
    self.pair_list = np.empty((0,self.mult),float)
    self.scale = jetpt

  def make_pairs(self):
    for i,part_i in enumerate(self.particle_list):
      inner_list = np.array([])
      for j,part_j in enumerate(self.particle_list[i+1:]):
        
        dist = calculate_distance(part_i,part_j)
        # print(' pairing particle i = ', i, ' and j = ', j+i+1, ' with distance ', dist)
        inner_list = np.append(inner_list, dist)

      inner_list = np.pad(inner_list, (i+1, 0), 'constant')
      inner_list.resize(1,self.mult)
      self.pair_list = np.append(self.pair_list,inner_list,axis=0)

      del inner_list

    # print(self.pair_list)

  def construct_EEC(self, hist):
    for ipart1 in range(self.mult):

      for ipart2 in range(ipart1+1,self.mult):
        dist12 = self.pair_list[ipart1][ipart2]
        # print(' EEC combining particle i =', ipart1, 'and j =', ipart2, 'with distance', dist12)
        
        # print(' E(i) =', self.particle_list[ipart1].E(), 'and E(j) =', self.particle_list[ipart2].E(), 'with jet pt =', self.scale)
        eec_weight = self.particle_list[ipart1].E()*self.particle_list[ipart2].E()/math.pow(self.scale,2)
        hist.Fill(self.scale, dist12, eec_weight) # scale is jet pt

  def construct_E3C(self, hist):
    for ipart1 in range(self.mult):

      for ipart2 in range(ipart1+1,self.mult):
        dist12 = self.pair_list[ipart1][ipart2]

        for ipart3 in range(ipart2+1,self.mult):
          dist23 = self.pair_list[ipart2][ipart3]
          dist13 = self.pair_list[ipart1][ipart3]

          dist_list= [dist12, dist23, dist13]
          dist_list_sorted = sorted(dist_list)
          dist_max = dist_list_sorted[len(dist_list)-1]
          # print(' E3C combining particle', ipart1, ipart2, ipart3, 'with distance', dist12, dist23, dist13,'max',dist_max)
        
          e3c_weight = self.particle_list[ipart1].E()*self.particle_list[ipart2].E()*self.particle_list[ipart3].E()/math.pow(self.scale,3)
          hist.Fill(self.scale, dist_max, e3c_weight)

  def construct_E4C(self, hist):
    for ipart1 in range(self.mult):

      for ipart2 in range(ipart1+1,self.mult):
        dist12 = self.pair_list[ipart1][ipart2]

        for ipart3 in range(ipart2+1,self.mult):
          dist23 = self.pair_list[ipart2][ipart3]
          dist13 = self.pair_list[ipart1][ipart3]

          for ipart4 in range(ipart3+1,self.mult):
            dist34 = self.pair_list[ipart3][ipart4]
            dist24 = self.pair_list[ipart2][ipart4]
            dist14 = self.pair_list[ipart1][ipart4]

            dist_list= [dist12, dist23, dist13, dist34, dist24, dist14]
            dist_list_sorted = sorted(dist_list)
            dist_max = dist_list_sorted[len(dist_list)-1]
        
            e4c_weight = self.particle_list[ipart1].E()*self.particle_list[ipart2].E()*self.particle_list[ipart3].E()*self.particle_list[ipart4].E()/math.pow(self.scale,4)
            hist.Fill(self.scale, dist_max, e4c_weight)

################################################################
class ProcessData_ENC(process_data_base.ProcessDataBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # print(sys.path)
    # print(sys.modules)

    # Initialize base class
    super(ProcessData_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]


  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):

    for jetR in self.jetR_list:
      for observable in self.observable_list:
        for trk_thrd in self.obs_settings[observable]:
            if self.is_pp:

                for ipoint in range(2, 5):
                    name = 'h_{}{}_JetPt_R{}_trk{}'.format(observable, ipoint, jetR, trk_thrd)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T,ch jet}')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)

                name = 'h_JetPt_R{}_trk{}'.format(jetR, trk_thrd)
                pt_bins = linbins(0,200,200)
                h = ROOT.TH1D(name, name, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('Counts')
                setattr(self, name, h)
                    
  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix):

    name = 'h_JetPt_R{}_trk{}'.format(jetR, obs_label) # FIX ME: this histogram does not need to be filled for every configuration and trk info is redundent (just there for the sake of unique names). 
    getattr(self, name).Fill(jet.perp())

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = []
    trk_thrd = obs_setting
    # print('jet R',jetR,'observable',observable,'thrd',trk_thrd)
    # print('const size',len(constituents))
    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted

      # print('jet',jet.perp(),'pt',c.pt())
    
    new_corr = CorrelatorBuilder(c_select,jet.perp())
    new_corr.make_pairs()
    for ipoint in range(2, 5):
        name = 'h_{}{}_JetPt_R{}_trk{}'.format(self.observable, ipoint, jetR, obs_label)
        if (ipoint==2):
            new_corr.construct_EEC(getattr(self, name))
        if (ipoint==3):
            new_corr.construct_E3C(getattr(self, name))
        if (ipoint==4):
            new_corr.construct_E4C(getattr(self, name))    
          

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process data')
  parser.add_argument('-f', '--inputFile', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='config/analysis_config.yaml',
                      help="Path of config file for analysis")
  parser.add_argument('-o', '--outputDir', action='store',
                      type=str, metavar='outputDir',
                      default='./TestOutput',
                      help='Output directory for output to be written to')
  
  # Parse the arguments
  args = parser.parse_args()
  
  print('Configuring...')
  print('inputFile: \'{0}\''.format(args.inputFile))
  print('configFile: \'{0}\''.format(args.configFile))
  print('ouputDir: \'{0}\"'.format(args.outputDir))
  print('----------------------------------------------------------------')
  
  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessData_ENC(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_data()
