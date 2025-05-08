#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of MC track information
  and do jet-finding, and save response histograms.
"""

from __future__ import print_function

# General
import os
import sys
import argparse

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
import array
import math
# from array import *

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjtools
import ecorrel

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_io_emb
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.alice_analysis.process.user.substructure import process_mc_base
from pyjetty.alice_analysis.process.base import thermal_generator
from pyjetty.mputils.csubtractor import CEventSubtractor

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class ProcessMC_ENC_Gen(process_mc_base.ProcessMCBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMC_ENC_Gen, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects_R(self, jetR):

    for observable in self.observable_list:

      for trk_thrd in self.obs_settings[observable]:

        obs_label = self.utils.obs_label(trk_thrd, None) 

        # Init ENC histograms (truth level only)
        if 'ENC' in observable:
          for ipoint in range(2, 3):
            # Truth histograms
            name = 'h_{}{}_JetPt_Truth_R{}_{}'.format(observable, ipoint, jetR, obs_label)
            pt_bins = linbins(0,200,200)
            RL_bins = logbins(1E-4,1,50)
            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
            h.GetXaxis().SetTitle('p_{T,ch jet}')
            h.GetYaxis().SetTitle('R_{L}')
            setattr(self, name, h)

            name = 'h_{}{}Pt_JetPt_Truth_R{}_{}'.format(observable, ipoint, jetR, obs_label) # pt scaled histograms (currently only for unmatched jets)
            pt_bins = linbins(0,200,200)
            ptRL_bins = logbins(1E-3,1E2,60)
            h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
            h.GetXaxis().SetTitle('p_{T,ch jet}')
            h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
            setattr(self, name, h)
        
        if 'jet_pt' in observable:
          name = 'h_{}_JetPt_Truth_R{}_{}'.format(observable, jetR, obs_label)
          pt_bins = linbins(0,200,200)
          h = ROOT.TH1D(name, name, 200, pt_bins)
          h.GetXaxis().SetTitle('p_{T,ch jet}')
          h.GetYaxis().SetTitle('Counts')
          setattr(self, name, h)

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # Fill 2D histogram of (pt, obs)
  #---------------------------------------------------------------
  def create_response_histograms(self, observable, ipoint, jetR, trk_thrd, R_max = None):
    return

  def is_same_charge(self, corr_builder, ipoint, constituents, index):
    part1 = int(corr_builder.correlator(ipoint).indices1()[index])
    part2 = int(corr_builder.correlator(ipoint).indices2()[index])
    q1 = int(constituents[part1].python_info().charge)
    q2 = int(constituents[part2].python_info().charge)

    if q1*q2 > 0:
      return True
    else:
      return False

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # Fill 2D histogram of (pt, obs)
  #---------------------------------------------------------------
  def fill_observable_histograms(self, hname, jet, jet_groomed_lund, jetR, obs_setting,
                                 grooming_setting, obs_label, jet_pt_ungroomed):
    # For ENC in PbPb, jet_pt_ungroomed stores the corrected jet pT
    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()
    trk_thrd = obs_setting

    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted
    
    dphi_cut = -9999
    deta_cut = -9999

    jet_pt = jet.perp()

    new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, 2, 1, dphi_cut, deta_cut)
    for observable in self.observable_list:
      if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
        for ipoint in range(2, 3):
          if self.smearRL:
              sigma = 0.001 # 1 mrad
              noise = np.random.normal(0, sigma, new_corr.correlator(ipoint).rs().size())
              #print("ipoint",ipoint,"thrd",trk_thrd,"noise",noise)
          for index in range(new_corr.correlator(ipoint).rs().size()):

            # processing only like-sign pairs when self.ENC_pair_like is on
            if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
              continue

            # processing only unlike-sign pairs when self.ENC_pair_unlike is on
            if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
              continue
            
            if 'ENC' in observable:
              if self.smearRL:
                  getattr(self, hname.format(observable + str(ipoint),obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index]+noise[index], new_corr.correlator(ipoint).weights()[index])
                  getattr(self, hname.format(observable + str(ipoint) + 'Pt',obs_label)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index]+noise[index], new_corr.correlator(ipoint).weights()[index])
              else:
                  getattr(self, hname.format(observable + str(ipoint),obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
                  getattr(self, hname.format(observable + str(ipoint) + 'Pt',obs_label)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

            if ipoint==2 and 'EEC_noweight' in observable:
              getattr(self, hname.format(observable,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])

            if ipoint==2 and 'EEC_weight2' in observable:
              getattr(self, hname.format(observable,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))

      if 'jet_pt' in observable:
        getattr(self, hname.format(observable,obs_label)).Fill(jet_pt)

  #---------------------------------------------------------------
  # This function is called per jet subconfigration 
  # Fill matched jet histograms
  #---------------------------------------------------------------
  def fill_matched_jet_histograms(self, jet_det, jet_det_groomed_lund, jet_truth,
                                  jet_truth_groomed_lund, jet_pp_det, jetR,
                                  obs_setting, grooming_setting, obs_label,
                                  jet_pt_det_ungroomed, jet_pt_truth_ungroomed, R_max, suffix, **kwargs):
    return
    
##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process MC')
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

  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessMC_ENC_Gen(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_mc()
