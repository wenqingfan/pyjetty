#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of MC track information
  and do jet-finding, and save response histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
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

ROOT.gSystem.Load("/global/cfs/cdirs/alice/wenqing/mypyjetty/heppy/external/roounfold/roounfold-current/lib/libRooUnfold.so")

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

from pyjetty.alice_analysis.analysis.base import common_base
from pyjetty.alice_analysis.analysis.base import analysis_utils

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class EEC_pair:
  def __init__(self, _index1, _index2, _weight, _r, _pt):
    self.index1 = _index1
    self.index2 = _index2
    self.weight = _weight
    self.r = _r
    self.pt = _pt

  def is_equal(self, pair2):
    return (self.index1 == pair2.index1 and self.index2 == pair2.index2) \
      or (self.index1 == pair2.index2 and self.index2 == pair2.index1)
  
  def __str__(self):
    return "EEC pair with (index1, index2, weight, RL, pt) = (" + \
      str(self.index1) + ", " + str(self.index2) + ", " + str(self.weight) + \
      ", " + str(self.r) + ", " + str(self.pt) + ")"

################################################################
class ProcessMC_ENC_2D(process_mc_base.ProcessMCBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMC_ENC_2D, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects_R(self, jetR):

    if self.is_pp:
      # define binnings
      # these are the truth level binnings
      n_bins_truth = [20, 7] # WARNING RooUnfold seg faults if too many bins used
      # these are the truth level binnings
      # binnings[0] -- weight, binnings[1] -- jet pT
      binnings_truth = [np.logspace(-5,0,n_bins_truth[0]+1), \
                  np.array([5, 10, 20, 40, 60, 80, 100, 150]).astype(float) ]
      # slight difference for reco pt bin
      # binnings[0] -- weight, binnings[1] -- jet pT
      n_bins_reco = [20, 6]
      binnings_reco = [np.logspace(-5,0,n_bins_reco[0]+1), \
                  np.array([10, 20, 40, 60, 80, 100, 150]).astype(float) ]
    else:
      # define binnings
      # these are the truth level binnings
      n_bins_truth = [20, 10] # WARNING RooUnfold seg faults if too many bins used
      # these are the truth level binnings
      # binnings[0] -- weight, binnings[1] -- jet pT
      binnings_truth = [np.logspace(-5,0,n_bins_truth[0]+1), \
                  np.array([20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]).astype(float) ]
      # slight difference for reco pt bin
      # binnings[0] -- weight, binnings[1] -- jet pT
      n_bins_reco = [20, 6]
      binnings_reco = [np.logspace(-5,0,n_bins_reco[0]+1), \
                  np.array([40, 50, 60, 70, 80, 100, 120]).astype(float) ]

    for observable in self.observable_list:
      
      # can take EEC with different energy power (currently only EEC with power n = 1 implemented)
      if observable == 'jet_ENC_RL':
      
        for trk_thrd in self.obs_settings[observable]:

          obs_label = self.utils.obs_label(trk_thrd, None) 

          #=======================================
          #      1D unfolding for jet pT
          #=======================================
          # 1D to 1D RM
          name = 'h_jetpt_reco1D_R{}_{}'.format(jetR, trk_thrd)
          h1_reco = ROOT.TH1D(name, name, n_bins_reco[1], binnings_reco[1])
          h1_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
          h1_reco.GetYaxis().SetTitle('Counts')
          setattr(self, name, h1_reco)
          name = 'h_jetpt_gen1D_R{}_{}'.format(jetR, trk_thrd)
          h1_gen = ROOT.TH1D(name, name, n_bins_truth[1], binnings_truth[1])
          h1_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
          h1_gen.GetYaxis().SetTitle('Counts')
          setattr(self, name, h1_gen)
          name = 'h_jetpt_response1D_R{}_{}'.format(jetR, trk_thrd)
          response1D = ROOT.RooUnfoldResponse(h1_reco, h1_gen)
          setattr(self, name, response1D)

          # efficiency and purity check (involving un-matched jets) is filled in process_mc_base.py

          #=======================================
          # 2D unfolding for energy correlators
          #=======================================
          name = 'h_{}_reco_R{}_{}'.format(observable, jetR, obs_label)
          h2_reco = ROOT.TH2D(name, name, n_bins_reco[0], binnings_reco[0], n_bins_reco[1], binnings_reco[1])
          h2_reco.GetXaxis().SetTitle('weight^{det}')
          h2_reco.GetZaxis().SetTitle('p^{det}_{T,ch jet}')
          setattr(self, name, h2_reco)
          name = 'h_{}_gen_R{}_{}'.format(observable, jetR, obs_label)
          h2_gen = ROOT.TH2D(name, name, n_bins_truth[0], binnings_truth[0], n_bins_truth[1], binnings_truth[1])
          h2_gen.GetXaxis().SetTitle('weight^{truth}')
          h2_gen.GetZaxis().SetTitle('p^{truth}_{T,ch jet}')
          setattr(self, name, h2_gen)
          name = 'h_{}_response_R{}_{}'.format(observable, jetR, obs_label)
          response = ROOT.RooUnfoldResponse(h2_reco, h2_gen)
          setattr(self, name, response)

          # for purity correction
          name = 'h_{}_reco_unmatched_R{}_{}'.format(observable, jetR, obs_label)
          h = ROOT.TH2D(name, name, n_bins_reco[0], binnings_reco[0], n_bins_reco[1], binnings_reco[1])
          h.GetXaxis().SetTitle('weight^{det}')
          h.GetZaxis().SetTitle('p^{det}_{T,ch jet}')
          setattr(self, name, h)          

  #---------------------------------------------------------------
  # This function is called per jet subconfigration 
  # Fill matched jet histograms
  #---------------------------------------------------------------
  def fill_matched_jet_histograms(self, jet_det, jet_det_groomed_lund, jet_truth,
                                  jet_truth_groomed_lund, jet_pp_det, jetR,
                                  obs_setting, grooming_setting, obs_label,
                                  jet_pt_det_ungroomed, jet_pt_truth_ungroomed, R_max, suffix, **kwargs):
    # assumes det and truth parts are matched beforehand:
    # matching particles are given matching user_index s
    # if some det or truth part does not have a match, it is given a unique index
    # also assumes all jet and particle level cuts have been applied already

    if self.do_rho_subtraction:
      jet_pt_det = jet_pt_det_ungroomed
    else:
      jet_pt_det = jet_det.perp()

    trk_thrd = obs_setting

    hname = 'h_jetpt_reco1D_R{}_{}'.format(jetR, obs_label)
    getattr(self, hname).Fill(jet_pt_det, self.pt_hat)
    hname = 'h_jetpt_gen1D_R{}_{}'.format(jetR, obs_label)
    getattr(self, hname).Fill(jet_truth.perp(), self.pt_hat)
    hname = 'h_jetpt_response1D_R{}_{}'.format(jetR, obs_label)
    getattr(self, hname).Fill(jet_pt_det, jet_truth.perp(), self.pt_hat)

    for observable in self.observable_list:
      
      if observable == 'jet_ENC_RL':
        
        # truth level EEC pairs
        truth_pairs = self.get_EEC_pairs(jet_truth, jet_truth.perp(), trk_thrd, ipoint=2)

        # det level EEC pairs
        det_pairs = self.get_EEC_pairs(jet_det, jet_pt_det, trk_thrd, ipoint=2)

        ######### purity correction #########
        # calculate det EEC cross section irregardless if truth match exists

        for d_pair in det_pairs:

          # fill one RL bin for now
          if d_pair.r > 0.04 and d_pair.r < 0.05:
            hname = 'h_{}_reco_unmatched_R{}_{}'.format(observable, jetR, obs_label)
            getattr(self, hname).Fill(d_pair.weight, d_pair.pt, self.pt_hat)

        ########################## TTree output generation #########################
        # composite of truth and smeared pairs, fill the TTree preprocessed
        dummyval = -9999

        # pair matching
        for t_pair in truth_pairs:

          # fill one RL bin for now
          if t_pair.r > 0.04 and t_pair.r < 0.05:
            hname = 'h_{}_gen_R{}_{}'.format(observable, jetR, obs_label)
            getattr(self, hname).Fill(t_pair.weight, t_pair.pt, self.pt_hat)

          match_found = False
          for d_pair in det_pairs:

            if d_pair.is_equal(t_pair):

              # fill one RL bin for now (assuming very similar d_pair.r and t_pair.r)
              if t_pair.r > 0.04 and t_pair.r < 0.05:
                hname = 'h_{}_reco_R{}_{}'.format(observable, jetR, obs_label)
                getattr(self, hname).Fill(d_pair.weight, d_pair.pt, self.pt_hat)
                hname = 'h_{}_response_R{}_{}'.format(observable, jetR, obs_label)
                getattr(self, hname).Fill(d_pair.weight, d_pair.pt, t_pair.weight, t_pair.pt, self.pt_hat)

                match_found = True
                break

          if not match_found:

            hname = 'h_{}_response_R{}_{}'.format(observable, jetR, obs_label)
            getattr(self, hname).Miss(t_pair.weight, t_pair.pt, self.pt_hat)
      
  #---------------------------------------------------------------
  # Return EEC pairs with the input threshold cut
  # NB: this is not the most efficient implementation 
  # when using multiple threshold cuts 
  #---------------------------------------------------------------
  def get_EEC_pairs(self, jet, jet_pt, trk_thrd, ipoint=2):

    pairs = []

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()

    for c in constituents:
      if c.pt() < trk_thrd:
        break # NB: use the break statement since constituents are already sorted (so make sure the constituents are sorted)
      if c.user_index() >= 0:
        c_select.append(c) # NB: only consider 'signal' particles
    
    if self.ENC_pair_cut and (not 'Truth' in hname):
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    # n-point correlator with all charged particles
    max_npoint = 2
    weight_power = 1
    cb = ecorrel.CorrelatorBuilder(c_select, jet_pt, max_npoint, weight_power, dphi_cut, deta_cut)

    EEC_cb = cb.correlator(ipoint)

    EEC_weights = EEC_cb.weights() # cb.correlator(npoint).weights() constains list of weights
    EEC_rs = EEC_cb.rs() # cb.correlator(npoint).rs() contains list of RL
    EEC_indicies1 = EEC_cb.indices1() # contains list of 1st track in the pair (index should be based on the indices in c_select)
    EEC_indicies2 = EEC_cb.indices2() # contains list of 2nd track in the pair

    for i in range(len(EEC_rs)):
      event_index1 = _v[EEC_indicies1[i]].user_index()
      event_index2 = _v[EEC_indicies2[i]].user_index()
      pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))

    return pairs
    

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

  analysis = ProcessMC_ENC_2D(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_mc()
