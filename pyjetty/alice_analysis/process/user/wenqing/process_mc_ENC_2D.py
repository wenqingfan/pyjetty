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
from bisect import bisect
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

  def jet_pair_type(self):
    # initialize to ss pair
    type1 = 1
    type2 = 1

    if self.index1 < -999:
      type1 = -1 # embed part
    else:
      type1 = 1  # pythia part
      # print('pythia part with >=0 index',self.index1) # double-check if the pythia part index is >=0

    if self.index2 < -999:
      type2 = -1 # embed part
    else:
      type2 = 1  # pythia part
      # print('pythia part with >=0 index',self.index2) # double-check if the pythia part index is >=0

    # NB: match the strings in self.pair_type_labels = ['_ss','_sb','_bb']
    if type1 < 0 and type2 < 0:
      # print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp()
      return 2 # means bkg-bkg
    if type1 < 0 and type2 > 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 < 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 > 0:
      # print('sig-sig (',type1,type2,') pt1',constituents[part1].perp()
      return 0 # means sig-sig

  def perpcone_pair_type(self):
    # initialize to ss pair
    type1 = 1
    type2 = 1

    if self.index1 < -999:
      type1 = 1 # jet part from embedding
    elif self.index1 >= 0:
      type1 = 1 # jet part from pythia
    else:
      type1 = -1 # perp part
      # print('perp part with -999 index',self.index1) # double-check if the perp part index is -999

    if self.index2 < -999:
      type2 = 1 # jet part from embedding
    elif self.index2 >= 0:
      type2 = 1 # jet part from pythia
    else:
      type2 = -1 # perp part
      # print('perp part with -999 index',self.index2) # double-check if the perp part index is -999

    # NB: match the strings in self.pair_type_labels = ['_ss','_sb','_bb']
    if type1 < 0 and type2 < 0:
      # print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp()
      return 2 # means bkg-bkg
    if type1 < 0 and type2 > 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 < 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 > 0:
      # print('sig-sig (',type1,type2,') pt1',constituents[part1].perp()
      return 0 # means sig-sig
  
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
      n_bins_truth = [20, 20] # WARNING RooUnfold seg faults if too many bins used
      # these are the truth level binnings
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      binnings_truth = [np.linspace(-5,0,n_bins_truth[0]+1), \
                  np.linspace(0,200,n_bins_truth[1]+1) ]
      # slight difference for reco pt bin
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      n_bins_reco = [20, 30]
      binnings_reco = [np.linspace(-5,0,n_bins_reco[0]+1), \
                  np.linspace(0,150,n_bins_reco[1]+1) ]

      self.n_RLbins = 25
      self.RLbins = logbins(1E-2,1,self.n_RLbins)
    else:
      # define binnings
      # these are the truth level binnings
      n_bins_truth = [20, 20] # WARNING RooUnfold seg faults if too many bins used
      # these are the truth level binnings
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      binnings_truth = [np.linspace(-5,0,n_bins_truth[0]+1), \
                  np.linspace(0,200,n_bins_truth[1]+1) ]
      # slight difference for reco pt bin
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      n_bins_reco = [20, 30]
      binnings_reco = [np.linspace(-5,0,n_bins_reco[0]+1), \
                  np.linspace(0,150,n_bins_reco[1]+1) ]

      self.n_RLbins = 25
      self.RLbins = logbins(1E-2,1,self.n_RLbins)

    self.pair_type_labels = ['']
    if self.do_rho_subtraction:
      self.pair_type_labels = ['_ss','_sb','_bb']
    if self.is_pp:
      self.pair_type_labels = ['_ss']

    for observable in self.observable_list:
      
      # can take EEC with different energy power (currently only EEC with power n = 1 implemented)
      if observable == 'jet_ENC_RL':
      
        for trk_thrd in self.obs_settings[observable]:

          obs_label = self.utils.obs_label(trk_thrd, None) 

          #=======================================
          #  jet or jetcone sigma_ENC histogram
          # and the corresponding jetcone hists
          #=======================================
          if not self.do_jetcone:
            ######### only signal pairs in jet ##########
            name = 'h_{}_sigma_reco_unmatched_R{}_{}'.format(observable, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetYaxis().SetTitle('R^{det}_{L}')
            h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
            setattr(self, name, h)

            name = 'h_{}_sigma_reco_matched_R{}_{}'.format(observable, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetYaxis().SetTitle('R^{det}_{L}')
            h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
            setattr(self, name, h)

            name = 'h_{}_sigma_gen_unmatched_R{}_{}'.format(observable, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetYaxis().SetTitle('R^{truth}_{L}')
            h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
            setattr(self, name, h)

            name = 'h_{}_sigma_gen_unmatched_kin_R{}_{}'.format(observable, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetYaxis().SetTitle('R^{truth}_{L}')
            h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
            setattr(self, name, h)

            name = 'h_{}_sigma_gen_matched_R{}_{}'.format(observable, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetYaxis().SetTitle('R^{truth}_{L}')
            h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
            setattr(self, name, h)

            ######### all pair types in jet ##########
            for pair_type_label in self.pair_type_labels:

              # histograms for unmatched jets (only filled for thermal closure)
              name = 'h_{}_sigma{}_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{det}_{L}')
              h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
              setattr(self, name, h)

              # histograms for matched jets
              name = 'h_{}_sigma{}_reco_unmatched_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{det}_{L}')
              h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
              setattr(self, name, h)

              name = 'h_{}_sigma{}_gen_unmatched_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{det}_{L}')
              h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h)

              name = 'h_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{det}_{L}')
              h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h)

            ######### all pair types in perpcone ##########
            if self.do_perpcone:  
              perpcone_R = jetR
              # one perpcone
              for pair_type_label in self.pair_type_labels:
                # histograms for unmatched jets (only filled for thermal closure)
                name = 'h_perpcone{}_{}_sigma{}_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h)

                # histograms for matched jets
                name = 'h_perpcone{}_{}_sigma{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_perpcone{}_{}_sigma{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_perpcone{}_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h)
              
              # two perpcones (only save perp1-perp2 pairs which correspond to 'sb')
              if self.do_2cones:
                # histograms for unmatched jets (only filled for thermal closure)
                name = 'h_2perpcone{}_{}_sigma{}_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h)

                # histograms for matched jets
                name = 'h_2perpcone{}_{}_sigma{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_2perpcone{}_{}_sigma{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_2perpcone{}_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h)
          
          else:
            for jetcone_R in self.jetcone_R_list:
              ######### only signal pairs in jetcone ##########
              name = 'h_jetcone{}_{}_sigma_reco_unmatched_R{}_{}'.format(jetcone_R, observable, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{det}_{L}')
              h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
              setattr(self, name, h)

              name = 'h_jetcone{}_{}_sigma_reco_matched_R{}_{}'.format(jetcone_R, observable, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{det}_{L}')
              h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
              setattr(self, name, h)

              name = 'h_jetcone{}_{}_sigma_gen_unmatched_R{}_{}'.format(jetcone_R, observable, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{truth}_{L}')
              h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h)

              name = 'h_jetcone{}_{}_sigma_gen_unmatched_kin_R{}_{}'.format(jetcone_R, observable, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{truth}_{L}')
              h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h)

              name = 'h_jetcone{}_{}_sigma_gen_matched_R{}_{}'.format(jetcone_R, observable, jetR, obs_label)
              pt_bins = linbins(0,200,40)
              h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
              h.GetYaxis().SetTitle('R^{truth}_{L}')
              h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h)

              ######### all pair types in jetcone ##########
              for pair_type_label in self.pair_type_labels:

                # histograms for unmatched jets (only filled for thermal closure)
                name = 'h_jetcone{}_{}_sigma{}_R{}_{}'.format(jetcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h)

                # histograms for matched jets
                name = 'h_jetcone{}_{}_sigma{}_reco_unmatched_R{}_{}'.format(jetcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_jetcone{}_{}_sigma{}_gen_unmatched_R{}_{}'.format(jetcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_jetcone{}_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(jetcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h)
              
              ######### all pair types in perpcone ##########
              if self.do_perpcone:  
                perpcone_R = jetcone_R
                # one perpcone
                for pair_type_label in self.pair_type_labels:
                  # histograms for unmatched jets (only filled for thermal closure)
                  name = 'h_perpcone{}_{}_sigma{}_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h)

                  # histograms for matched jets
                  name = 'h_perpcone{}_{}_sigma{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h)

                  name = 'h_perpcone{}_{}_sigma{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h)

                  name = 'h_perpcone{}_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h)

                # two perpcones
                if self.do_2cones:
                  # histograms for unmatched jets (only filled for thermal closure)
                  name = 'h_2perpcone{}_{}_sigma{}_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h)

                  # histograms for matched jets
                  name = 'h_2perpcone{}_{}_sigma{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h)

                  name = 'h_2perpcone{}_{}_sigma{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h)

                  name = 'h_2perpcone{}_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, '_sb', jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetYaxis().SetTitle('R^{det}_{L}')
                  h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h)
          
          #=======================================
          #      1D unfolding for jet pT
          #=======================================
          # 1D to 1D RM
          name = 'h_jetpt_reco1D_matched_R{}_{}'.format(jetR, obs_label)
          h1_reco = ROOT.TH1D(name, name, n_bins_reco[1], binnings_reco[1])
          h1_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
          h1_reco.GetYaxis().SetTitle('Counts')
          setattr(self, name, h1_reco)

          name = 'h_jetpt_gen1D_matched_R{}_{}'.format(jetR, obs_label)
          h1_gen = ROOT.TH1D(name, name, n_bins_truth[1], binnings_truth[1])
          h1_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
          h1_gen.GetYaxis().SetTitle('Counts')
          setattr(self, name, h1_gen)

          name = 'h_jetpt_gen1D_matched_kin_R{}_{}'.format(jetR, obs_label)
          h1_gen_kin = ROOT.TH1D(name, name, n_bins_truth[1], binnings_truth[1])
          h1_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
          h1_gen_kin.GetYaxis().SetTitle('Counts')
          setattr(self, name, h1_gen_kin)
          
          if self.save_RUResponse:
            # save response matrix in RooUnfoldResponse format directly
            name = 'h_jetpt_response1D_R{}_{}'.format(jetR, obs_label)
            response1D = ROOT.RooUnfoldResponse(h1_reco, h1_gen)
            response1D.SetName(name)
            setattr(self, name, response1D)
          else:
            # save response matrix in THnF format
            dim = 2
            title = ['p^{det}_{T,ch jet}', 'p^{truth}_{T,ch jet}']
            nbins = [30, 20]
            min = [0., 0.]
            max = [150., 200.]
            name = 'THnF_jetpt_response1D_R{}_{}'.format(jetR, obs_label)
            self.create_thn(name, title, dim, nbins, min, max)

          # efficiency and purity check (involving un-matched jets) is filled in process_mc_base.py

          #=======================================
          # 2D unfolding for energy correlators
          # either for jet constituents or 
          # jet cone particles
          # and the corresponding perpcone
          #=======================================
          for iRL in range(self.n_RLbins):
            if not self.do_jetcone:
              
              ######### only signal pairs in jet ##########
              name = 'h_{}{:d}_reco_matched_R{}_{}'.format(observable, iRL, jetR, obs_label)
              h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
              h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
              h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
              setattr(self, name, h2_reco)
              
              name = 'h_{}{:d}_gen_matched_R{}_{}'.format(observable, iRL, jetR, obs_label)
              h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
              h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
              h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h2_gen)

              name = 'h_{}{:d}_gen_unmatched_R{}_{}'.format(observable, iRL, jetR, obs_label)
              h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
              h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
              h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h2_gen)
              
              # histogram to study the kinematic efficiency effect due to truncation of det jet pt
              name = 'h_{}{:d}_gen_unmatched_kin_R{}_{}'.format(observable, iRL, jetR, obs_label)
              h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
              h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
              h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h2_gen_kin)

              ######### all pair types in jet ##########
              for pair_type_label in self.pair_type_labels:

                # histograms for unmatched jets (only filled for thermal closure)
                name = 'h_{}{:d}{}_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
                h2_raw = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                h2_raw.GetYaxis().SetTitle('log10(weight^{det})')
                h2_raw.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h2_raw)

                # histograms for matched jets
                name = 'h_{}{:d}{}_reco_unmatched_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
                h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h2_reco)

                name = 'h_{}{:d}{}_gen_unmatched_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
                h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h2_gen)

                name = 'h_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
                h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
                h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h2_gen_kin)

              ######### all pair types in perpcone ##########
              if self.do_perpcone:
                perpcone_R = jetR
                # one perpcone
                for pair_type_label in self.pair_type_labels:
            
                  # histograms for unmatched jets (only filled for thermal closure)
                  name = 'h_perpcone{}_{}{:d}{}_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_raw = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_raw.GetYaxis().SetTitle('log10(weight^{det})')
                  h2_raw.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h2_raw)

                  # histograms for matched jets
                  name = 'h_perpcone{}_{}{:d}{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                  h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h2_reco)

                  name = 'h_perpcone{}_{}{:d}{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                  h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                  h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h2_gen)

                  name = 'h_perpcone{}_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                  h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
                  h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h2_gen_kin)

                # two perpcones
                if self.do_2cones:
                  # histograms for unmatched jets (only filled for thermal closure)
                  name = 'h_2perpcone{}_{}{:d}{}_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                  h2_raw = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_raw.GetYaxis().SetTitle('log10(weight^{det})')
                  h2_raw.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h2_raw)

                  # histograms for matched jets
                  name = 'h_2perpcone{}_{}{:d}{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                  h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                  h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h2_reco)

                  name = 'h_2perpcone{}_{}{:d}{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                  h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                  h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                  h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h2_gen)

                  name = 'h_2perpcone{}_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                  h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                  h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
                  h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h2_gen_kin)
          
            else:
              for jetcone_R in self.jetcone_R_list:

                ######### only signal pairs in jetcone ##########
                name = 'h_jetcone{}_{}{:d}_reco_matched_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h2_reco)
                
                name = 'h_jetcone{}_{}{:d}_gen_matched_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h2_gen)

                name = 'h_jetcone{}_{}{:d}_gen_unmatched_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h2_gen)
                
                # histogram to study the kinematic efficiency effect due to truncation of det jet pt
                name = 'h_jetcone{}_{}{:d}_gen_unmatched_kin_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
                h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h2_gen_kin)

                ######### all pair types in jetcone ##########
                for pair_type_label in self.pair_type_labels:

                  # histograms for unmatched jets (only filled for thermal closure)
                  name = 'h_jetcone{}_{}{:d}{}_R{}_{}'.format(jetcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_raw = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_raw.GetYaxis().SetTitle('log10(weight^{det})')
                  h2_raw.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h2_raw)

                  # histograms for matched jets
                  name = 'h_jetcone{}_{}{:d}{}_reco_unmatched_R{}_{}'.format(jetcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                  h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h2_reco)

                  name = 'h_jetcone{}_{}{:d}{}_gen_unmatched_R{}_{}'.format(jetcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                  h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                  h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h2_gen)

                  name = 'h_jetcone{}_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(jetcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                  h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
                  h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h2_gen_kin)
                
                ######### all pair types in perpcone ##########
                if self.do_perpcone:
                  perpcone_R = jetcone_R
                  # one perpcone
                  for pair_type_label in self.pair_type_labels:
                    # histograms for unmatched jets (only filled for thermal closure)
                    name = 'h_perpcone{}_{}{:d}{}_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    h2_raw = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                    h2_raw.GetYaxis().SetTitle('log10(weight^{det})')
                    h2_raw.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                    setattr(self, name, h2_raw)

                    # histograms for matched jets
                    name = 'h_perpcone{}_{}{:d}{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                    h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                    h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                    setattr(self, name, h2_reco)

                    name = 'h_perpcone{}_{}{:d}{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                    h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                    h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                    setattr(self, name, h2_gen)

                    name = 'h_perpcone{}_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                    h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
                    h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                    setattr(self, name, h2_gen_kin)

                  # two perpcones
                  if self.do_2cones:
                    # histograms for unmatched jets (only filled for thermal closure)
                    name = 'h_2perpcone{}_{}{:d}{}_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                    h2_raw = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                    h2_raw.GetYaxis().SetTitle('log10(weight^{det})')
                    h2_raw.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                    setattr(self, name, h2_raw)

                    # histograms for matched jets
                    name = 'h_2perpcone{}_{}{:d}{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                    h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                    h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                    h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                    setattr(self, name, h2_reco)

                    name = 'h_2perpcone{}_{}{:d}{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                    h2_gen = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                    h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                    h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                    setattr(self, name, h2_gen)

                    name = 'h_2perpcone{}_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                    h2_gen_kin = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                    h2_gen_kin.GetYaxis().SetTitle('log10(weight^{truth})')
                    h2_gen_kin.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                    setattr(self, name, h2_gen_kin)
                      
          #============================================
          # Response matrix and purity hists for EEC
          #============================================
          for iRL in range(self.n_RLbins):
            if not self.do_jetcone: 
              # RMs     
              if self.save_RUResponse:
                ######### only signal pairs in jet ##########
                # save response matrix in RooUnfoldResponse format directly
                # fill misses pair by pair
                name = 'h_{}{:d}_response_R{}_{}'.format(observable, iRL, jetR, obs_label)
                response = ROOT.RooUnfoldResponse(h2_reco, h2_gen)
                response.SetName(name)
                setattr(self, name, response)
              else:
                ######### only signal pairs in jet ##########
                # save response matrix in THnF format
                dim = 4
                title = ['p^{det}_{T,ch jet}', 'p^{truth}_{T,ch jet}', 'log10(weight^{det})', 'log10(weight^{truth})']
                nbins = [30, 20, 20, 20]
                min = [0., 0., -5., -5.]
                max = [150., 200., 0., 0.]
                name = 'THnF_{}{:d}_response_R{}_{}'.format(observable, iRL, jetR, obs_label)
                self.create_thn(name, title, dim, nbins, min, max)

                # fill misses in separate histograms
                dim = 3
                # save the matched det jet pt for study of kinematic effect later
                title = ['p^{truth}_{T,ch jet}', 'log10(weight^{truth})', 'p^{det}_{T,ch jet}']
                nbins = [20, 20, 30]
                min = [0., -5., 0.]
                max = [200., 0., 150.]
                name = 'THnF_{}{:d}_response_miss_R{}_{}'.format(observable, iRL, jetR, obs_label)
                self.create_thn(name, title, dim, nbins, min, max)

                ######### all pair types in jet ##########
                dim = 4
                title = ['p^{det}_{T,ch jet}', 'p^{truth}_{T,ch jet}', 'log10(weight^{det})', 'log10(weight^{truth})']
                nbins = [30, 20, 20, 20]
                min = [0., 0., -5., -5.]
                max = [150., 200., 0., 0.]
                for pair_type_label in self.pair_type_labels:
                  name = 'THnF_{}{:d}{}_response_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
                  self.create_thn(name, title, dim, nbins, min, max)

                ######### all pair types in perpcone ##########
                if self.do_perpcone:
                  perpcone_R = jetR
                  # one cone
                  for pair_type_label in self.pair_type_labels:
                    name = 'THnF_perpcone{}_{}{:d}{}_response_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    self.create_thn(name, title, dim, nbins, min, max)
                  # two cones
                  if self.do_2cones:
                    name = 'THnF_2perpcone{}_{}{:d}{}_response_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                    self.create_thn(name, title, dim, nbins, min, max)

              # for purity correction
              name = 'h_{}{:d}_reco_unmatched_R{}_{}'.format(observable, iRL, jetR, obs_label)
              h = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
              h.GetYaxis().SetTitle('log10(weight^{det})')
              h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
              setattr(self, name, h) 

              # for purity correction after unfolding
              name = 'h_{}{:d}_ss_reco_matched_R{}_{}'.format(observable, iRL, jetR, obs_label)
              h = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
              h.GetYaxis().SetTitle('log10(weight^{truth})')
              h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h) 

            else:
              for jetcone_R in self.jetcone_R_list:
                # RMs
                if self.save_RUResponse:
                  ######### only signal pairs in jetcone ##########
                  # save response matrix in RooUnfoldResponse format directly
                  # fill misses pair by pair
                  name = 'h_jetcone{}_{}{:d}_response_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                  response = ROOT.RooUnfoldResponse(h2_reco, h2_gen)
                  response.SetName(name)
                  setattr(self, name, response)
                else:
                  ######### only signal pairs in jetcone ##########
                  # save response matrix in THnF format
                  dim = 4
                  title = ['p^{det}_{T,ch jet}', 'p^{truth}_{T,ch jet}', 'log10(weight^{det})', 'log10(weight^{truth})']
                  nbins = [30, 20, 20, 20]
                  min = [0., 0., -5., -5.]
                  max = [150., 200., 0., 0.]
                  name = 'THnF_jetcone{}_{}{:d}_response_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                  self.create_thn(name, title, dim, nbins, min, max)

                  # fill misses in separate histograms
                  dim = 3
                  # save the matched det jet pt for study of kinematic effect later
                  title = ['p^{truth}_{T,ch jet}', 'log10(weight^{truth})', 'p^{det}_{T,ch jet}']
                  nbins = [20, 20, 30]
                  min = [0., -5., 0.]
                  max = [200., 0., 150.]
                  name = 'THnF_jetcone{}_{}{:d}_response_miss_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                  self.create_thn(name, title, dim, nbins, min, max)

                  ######### all pair types in jetcone ##########
                  dim = 4
                  title = ['p^{det}_{T,ch jet}', 'p^{truth}_{T,ch jet}', 'log10(weight^{det})', 'log10(weight^{truth})']
                  nbins = [30, 20, 20, 20]
                  min = [0., 0., -5., -5.]
                  max = [150., 200., 0., 0.]
                  for pair_type_label in self.pair_type_labels:
                    name = 'THnF_jetcone{}_{}{:d}{}_response_R{}_{}'.format(jetcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    self.create_thn(name, title, dim, nbins, min, max)

                  ######### all pair types in perpcone ##########
                  if self.do_perpcone:
                    perpcone_R = jetcone_R
                    # one cone
                    for pair_type_label in self.pair_type_labels:
                      name = 'THnF_perpcone{}_{}{:d}{}_response_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                      self.create_thn(name, title, dim, nbins, min, max)
                    # two cones
                    if self.do_2cones:
                      name = 'THnF_2perpcone{}_{}{:d}{}_response_R{}_{}'.format(perpcone_R, observable, iRL, '_sb', jetR, obs_label)
                      self.create_thn(name, title, dim, nbins, min, max)

                # for purity correction
                name = 'h_jetcone{}_{}{:d}_reco_unmatched_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                h = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                h.GetYaxis().SetTitle('log10(weight^{det})')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h) 

                # for purity correction after unfolding
                name = 'h_jetcone{}_{}{:d}_ss_reco_matched_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                h = ROOT.TH2D(name, name, n_bins_truth[1], binnings_truth[1], n_bins_truth[0], binnings_truth[0])
                h.GetYaxis().SetTitle('log10(weight^{truth})')
                h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                setattr(self, name, h) 
          
          #=======================================
          # RL resolution check for pairs
          #=======================================
          if not self.do_jetcone:
            name = 'h2d_matched_pair_RL_truth_vs_det_R{}_{}'.format(jetR, obs_label)
            h = ROOT.TH2D(name, name, self.n_RLbins, self.RLbins, self.n_RLbins, self.RLbins)
            h.GetXaxis().SetTitle('R_{L}^{det}')
            h.GetYaxis().SetTitle('R_{L}^{truth}')
            setattr(self, name, h)  
          else:
            for jetcone_R in self.jetcone_R_list:
              name = 'h2d_jetcone{}_matched_pair_RL_truth_vs_det_R{}_{}'.format(jetcone_R, jetR, obs_label)
              h = ROOT.TH2D(name, name, self.n_RLbins, self.RLbins, self.n_RLbins, self.RLbins)
              h.GetXaxis().SetTitle('R_{L}^{det}')
              h.GetYaxis().SetTitle('R_{L}^{truth}')
              setattr(self, name, h)  
  
  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # Fill 2D histogram of (pt, obs)
  # This is just a dummy function here
  #---------------------------------------------------------------
  def fill_observable_histograms(self, hname, jet, jet_groomed_lund, jetR, obs_setting,
                                 grooming_setting, obs_label, jet_pt_ungroomed, **kwargs):
    return

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # EEC for unmatched jets (used in the thermal closure test)
  # only filled for the det-level histograms
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix):

    trk_thrd = obs_setting

    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      if jet.area() == 0:
        return # NB: skip the zero area jets for now (also skip the perp-cone and jet-cone w.r.t. the zero area jets)
    else:
      jet_pt = jet.perp()
    # print('unsubtracted pt',jet.perp(),'subtracted',jet_pt,'# of constituents >',trk_thrd,'is',len(c_select))

    for observable in self.observable_list:

      if observable == 'jet_ENC_RL':
        cone_parts = None # no cone parts when looking at the jet EEC
        pairs_all = self.get_EEC_pairs(jet, cone_parts, jet_pt, trk_thrd, ipoint=2, only_signal_pairs=False)
        for pair in pairs_all:
          pair_type = pair.jet_pair_type() 
          pair_type_label = self.pair_type_labels[pair_type]

          RL = pair.r
          weight = pair.weight
          
          hname = 'h_{}_sigma{}_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
          getattr(self, hname).Fill(jet_pt, RL, weight)

          # determine RL bin for det pairs
          iRL = bisect(self.RLbins, RL)-1 # index from 0

          if iRL >= 0 and iRL < self.n_RLbins:
            hname = 'h_{}{:d}{}_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(jet_pt, np.log10(weight))
          
  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # EEC for unmatched jets (used in the thermal closure test)
  # only filled for the det-level histograms
  #---------------------------------------------------------------
  def fill_jetcone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix):

    trk_thrd = obs_setting

    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      if jet.area() == 0:
        return # NB: skip the zero area jets for now (also skip the perp-cone and jet-cone w.r.t. the zero area jets)
    else:
      jet_pt = jet.perp()

    for observable in self.observable_list:

      if observable == 'jet_ENC_RL':
        pairs_all = self.get_EEC_pairs(jet, cone_parts, jet_pt, trk_thrd, ipoint=2, only_signal_pairs=False)
        for pair in pairs_all:
          pair_type = pair.jet_pair_type() 
          pair_type_label = self.pair_type_labels[pair_type]

          RL = pair.r
          weight = pair.weight
          
          hname = 'h_jetcone{}_{}_sigma{}_R{}_{}'.format(cone_R, observable, pair_type_label, jetR, obs_label)
          getattr(self, hname).Fill(jet_pt, RL, weight)

          # determine RL bin for det pairs
          iRL = bisect(self.RLbins, RL)-1 # index from 0

          if iRL >= 0 and iRL < self.n_RLbins:
            hname = 'h_jetcone{}_{}{:d}{}_R{}_{}'.format(cone_R, observable, iRL, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(jet_pt, np.log10(weight))

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # EEC for unmatched jets (used in the thermal closure test)
  # only filled for the det-level histograms
  #---------------------------------------------------------------
  def fill_perpcone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix):

    # combine sig jet and perp cone with trk threshold cut
    trk_thrd = obs_setting

    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      if jet.area() == 0:
        return # NB: skip the zero area jets for now (also skip the perp-cone and jet-cone w.r.t. the zero area jets)
    else:
      jet_pt = jet.perp()
    
    for observable in self.observable_list:

      if observable == 'jet_ENC_RL':

        pairs_all = self.get_perpcone_EEC_pairs(cone_parts, jet_pt, trk_thrd, ipoint=2) 

        for pair in pairs_all:
          pair_type = pair.perpcone_pair_type() 
          pair_type_label = self.pair_type_labels[pair_type]

          RL = pair.r
          weight = pair.weight
          
          hname = 'h_perpcone{}_{}_sigma{}_R{}_{}'.format(cone_R, observable, pair_type_label, jetR, obs_label)
          getattr(self, hname).Fill(jet_pt, RL, weight)

          # determine RL bin for det pairs
          iRL = bisect(self.RLbins, RL)-1 # index from 0

          if iRL >= 0 and iRL < self.n_RLbins:
            hname = 'h_perpcone{}_{}{:d}{}_R{}_{}'.format(cone_R, observable, iRL, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(jet_pt, np.log10(weight))

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # EEC for unmatched jets (used in the thermal closure test)
  # only filled for the det-level histograms
  #---------------------------------------------------------------
  def fill_2perpcone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix):

    # combine sig jet and perp cone with trk threshold cut
    trk_thrd = obs_setting

    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      if jet.area() == 0:
        return # NB: skip the zero area jets for now (also skip the perp-cone and jet-cone w.r.t. the zero area jets)
    else:
      jet_pt = jet.perp()
    
    for observable in self.observable_list:

      if observable == 'jet_ENC_RL':

        pairs_all = self.get_perpcone_EEC_pairs(cone_parts, jet_pt, trk_thrd, ipoint=2) 

        for pair in pairs_all:
          pair_type = pair.perpcone_pair_type() 
          pair_type_label = self.pair_type_labels[pair_type]

          # for two perpcones, only save the perp1-perp2 pairs
          if pair_type_label != '_sb':
            continue

          RL = pair.r
          weight = pair.weight
          
          hname = 'h_2perpcone{}_{}_sigma{}_R{}_{}'.format(cone_R, observable, pair_type_label, jetR, obs_label)
          getattr(self, hname).Fill(jet_pt, RL, weight)

          # determine RL bin for det pairs
          iRL = bisect(self.RLbins, RL)-1 # index from 0

          if iRL >= 0 and iRL < self.n_RLbins:
            hname = 'h_2perpcone{}_{}{:d}{}_R{}_{}'.format(cone_R, observable, iRL, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(jet_pt, np.log10(weight))

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

    if self.jetscape:
      holes_in_det_jet = kwargs['holes_in_det_jet']
      holes_in_truth_jet = kwargs['holes_in_truth_jet']

    cone_parts_in_det_jet = kwargs['cone_parts_in_det_jet']
    cone_parts_in_truth_jet = kwargs['cone_parts_in_truth_jet']
    cone_R = kwargs['cone_R']
    cone_label = kwargs['cone_label']

    if self.do_rho_subtraction:
      jet_pt_det = jet_pt_det_ungroomed
    else:
      jet_pt_det = jet_det.perp()

    trk_thrd = obs_setting

    for observable in self.observable_list:
      
      if observable == 'jet_ENC_RL':

        ####### Fill jet pt histogram only once for each jet R #######
        # to avoid double/triple-counting when perpcone or jetcone is enabled
        if (cone_label == '') or (cone_label == '_jetcone{}'.format(self.jetcone_R_list[0])):
          hname = 'h_jetpt_reco1D_matched_R{}_{}'.format(jetR, obs_label)
          getattr(self, hname).Fill(jet_pt_det)
          hname = 'h_jetpt_gen1D_matched_R{}_{}'.format(jetR, obs_label)
          getattr(self, hname).Fill(jet_truth.perp())
          if jet_pt_det > 40:
            hname = 'h_jetpt_gen1D_matched_kin_R{}_{}'.format(jetR, obs_label)
            getattr(self, hname).Fill(jet_truth.perp())
          
          if self.save_RUResponse:
            hname = 'h_jetpt_response1D_R{}_{}'.format(jetR, obs_label)
            getattr(self, hname).Fill(jet_pt_det, jet_truth.perp(), self.pt_hat)
          else:
            hname = 'THnF_jetpt_response1D_R{}_{}'.format(jetR, obs_label)
            getattr(self, hname).Fill(jet_pt_det, jet_truth.perp())
        
        # type 1 -- fill EEC for jet constituents or jet cone parts
        if cone_label == '' or cone_label=='_jetcone{}'.format(cone_R):
            
          ################### all pair types ###################
          det_pairs_all = self.get_EEC_pairs(jet_det, cone_parts_in_det_jet, jet_pt_det, trk_thrd, ipoint=2, only_signal_pairs=False)

          for d_pair in det_pairs_all:
            pair_type = d_pair.jet_pair_type() 
            pair_type_label = self.pair_type_labels[pair_type]

            hname = 'h{}_{}_sigma{}_reco_unmatched_R{}_{}'.format(cone_label, observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

            t_pair_pt = jet_truth.pt()
            t_pair_weight = d_pair.weight*d_pair.pt*d_pair.pt/(t_pair_pt*t_pair_pt) # replace det jet pt by truth jet pt in the energy weight calculation

            hname = 'h{}_{}_sigma{}_gen_unmatched_R{}_{}'.format(cone_label, observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(t_pair_pt, d_pair.r, t_pair_weight) # fill det RL (assuming very similar det RL and truth RL)

            if jet_pt_det > 40:
              hname = 'h{}_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(cone_label, observable, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(t_pair_pt, d_pair.r, t_pair_weight) # fill det RL (assuming very similar det RL and truth RL)

            # determine RL bin for det pairs
            iRL = bisect(self.RLbins, d_pair.r)-1 # index from 0

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h{}_{}{:d}{}_reco_unmatched_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight)) 

              hname = 'h{}_{}{:d}{}_gen_unmatched_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(t_pair_pt, np.log10(t_pair_weight)) 

              if jet_pt_det > 40:
                hname = 'h{}_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
                getattr(self, hname).Fill(t_pair_pt, np.log10(t_pair_weight)) 

              if not self.save_RUResponse:
                hname = 'THnF{}_{}{:d}{}_response_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
                x = ([d_pair.pt, t_pair_pt, np.log10(d_pair.weight), np.log10(t_pair_weight)])
                x_array = array.array('d', x)
                getattr(self, hname).Fill(x_array)

          
          ################### Only signal pairs ###################
          # truth level EEC pairs
          truth_pairs = self.get_EEC_pairs(jet_truth, cone_parts_in_truth_jet, jet_truth.perp(), trk_thrd, ipoint=2, only_signal_pairs=True)

          # det level EEC pairs (only ss pairs)
          det_pairs = self.get_EEC_pairs(jet_det, cone_parts_in_det_jet, jet_pt_det, trk_thrd, ipoint=2, only_signal_pairs=True)

          ######### purity correction #########
          # calculate det EEC cross section irregardless if truth match exists

          for d_pair in det_pairs:

            hname = 'h{}_{}_sigma_reco_unmatched_R{}_{}'.format(cone_label, observable, jetR, obs_label)
            getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

            # determine RL bin for det pairs
            iRL = bisect(self.RLbins, d_pair.r)-1 # index from 0

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h{}_{}{:d}_reco_unmatched_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
              getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight))

          ########################## Fill RM (matched and missed) #########################
          dummyval = -9999

          # pair matching
          for t_pair in truth_pairs:

            hname = 'h{}_{}_sigma_gen_unmatched_R{}_{}'.format(cone_label, observable, jetR, obs_label)
            getattr(self, hname).Fill(t_pair.pt, t_pair.r, t_pair.weight)

            if jet_pt_det > 40:
              hname = 'h{}_{}_sigma_gen_unmatched_kin_R{}_{}'.format(cone_label, observable, jetR, obs_label)
              getattr(self, hname).Fill(t_pair.pt, t_pair.r, t_pair.weight)

            # determine RL bin fr truth pairs
            iRL = bisect(self.RLbins, t_pair.r)-1 # index from 0

            # if iRL ==40:
            #   print("new trurh pair")

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h{}_{}{:d}_gen_unmatched_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
              getattr(self, hname).Fill(t_pair.pt, np.log10(t_pair.weight))

              # kinematic effect because of truncation in matched det pT
              if jet_pt_det > 40: # hard code the truncation value as 40 for now
                hname = 'h{}_{}{:d}_gen_unmatched_kin_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
                getattr(self, hname).Fill(t_pair.pt, np.log10(t_pair.weight))
              # if iRL == 40:
              #   print('gen pair with distance',t_pair.r,'weight',t_pair.weight,'pt',t_pair.pt)

            match_found = False
            for d_pair in det_pairs:

              if d_pair.is_equal(t_pair):

                hname = 'h{}_{}_sigma_gen_matched_R{}_{}'.format(cone_label, observable, jetR, obs_label)
                getattr(self, hname).Fill(t_pair.pt, t_pair.r, t_pair.weight)

                hname = 'h{}_{}_sigma_reco_matched_R{}_{}'.format(cone_label, observable, jetR, obs_label)
                getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

                # fill the RL at det v.s. truth level (no energy weight)
                hname = 'h2d{}_matched_pair_RL_truth_vs_det_R{}_{}'.format(cone_label, jetR, obs_label)
                getattr(self, hname).Fill(d_pair.r, t_pair.r)
                
                # if iRL == 40:
                #   print('matched reco pair with distance',d_pair.r,'weight',d_pair.weight,'pt',d_pair.pt)

                # NB: assuming very similar d_pair.r and t_pair.r
                if iRL >= 0 and iRL < self.n_RLbins:

                  hname = 'h{}_{}{:d}_gen_matched_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
                  getattr(self, hname).Fill(t_pair.pt, np.log10(t_pair.weight))

                  hname = 'h{}_{}{:d}_reco_matched_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
                  getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight))

                  # this is for purity correction after unfolding and bkg subtraction
                  # NB1: two ways to fill weight here: 1. use det particle pT; 2. use truth particle pT (meaning truth for both jet pT and particle pT)
                  # NB2: one other implicit assumption is small migration for RL (in fact iRL used here is determined from truth RL)
                  # FIX ME: option 2 mention in NB1 is hard coded here
                  hname = 'h{}_{}{:d}_ss_reco_matched_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
                  getattr(self, hname).Fill(t_pair.pt, np.log10(t_pair.weight))
                  
                  if self.save_RUResponse:
                    hname = 'h{}_{}{:d}_response_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label, self.pt_hat) # NB: if RooUnfoldResponse format, applying scaling during while processing
                    getattr(self, hname).Fill(d_pair.pt, d_pair.weight, t_pair.pt, t_pair.weight)
                  else:
                    hname = 'THnF{}_{}{:d}_response_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
                    x = ([d_pair.pt, t_pair.pt, np.log10(d_pair.weight), np.log10(t_pair.weight)])
                    x_array = array.array('d', x)
                    getattr(self, hname).Fill(x_array)

                match_found = True
                break

            if not match_found:

              # if iRL == 40:
              #   print('unmatched reco pair with distance',d_pair.r,'weight',d_pair.weight,'pt',d_pair.pt)

              if iRL >= 0 and iRL < self.n_RLbins:
                
                if self.save_RUResponse:
                  hname = 'h{}_{}{:d}_response_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
                  getattr(self, hname).Miss(t_pair.pt, t_pair.weight, self.pt_hat)  # NB: if RooUnfoldResponse format, applying scaling during while processing
                else:
                  hname = 'THnF{}_{}{:d}_response_miss_R{}_{}'.format(cone_label, observable, iRL, jetR, obs_label)
                  x = ([t_pair.pt, np.log10(t_pair.weight), jet_pt_det])
                  x_array = array.array('d', x)
                  getattr(self, hname).Fill(x_array)

        # type 2 -- fill for perp cone or two perpcones
        if cone_label == '_perpcone{}'.format(cone_R) or cone_label == '_2perpcone{}'.format(cone_R): 

          ################### all pair types ###################
          det_pairs_all = self.get_perpcone_EEC_pairs(cone_parts_in_det_jet, jet_pt_det, trk_thrd, ipoint=2) 

          # FIX ME: what about the purity effect?
          for d_pair in det_pairs_all:
            pair_type = d_pair.perpcone_pair_type() 
            pair_type_label = self.pair_type_labels[pair_type]
            # print('pair index1', d_pair.index1, 'index2', d_pair.index2, 'pair type', pair_type, 'label', pair_type_label)

            if cone_label == '_2perpcone{}'.format(cone_R) and pair_type_label != '_sb':
              continue

            hname = 'h{}_{}_sigma{}_reco_unmatched_R{}_{}'.format(cone_label, observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

            t_pair_pt = jet_truth.pt()
            t_pair_weight = d_pair.weight*d_pair.pt*d_pair.pt/(t_pair_pt*t_pair_pt) # replace det jet pt by truth jet pt in the energy weight calculation

            hname = 'h{}_{}_sigma{}_gen_unmatched_R{}_{}'.format(cone_label, observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(t_pair_pt, d_pair.r, t_pair_weight) # fill det RL (assuming very similar det RL and truth RL)

            if jet_pt_det:
              hname = 'h{}_{}_sigma{}_gen_unmatched_kin_R{}_{}'.format(cone_label, observable, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(t_pair_pt, d_pair.r, t_pair_weight) # fill det RL (assuming very similar det RL and truth RL)

            # determine RL bin for det pairs
            iRL = bisect(self.RLbins, d_pair.r)-1 # index from 0

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h{}_{}{:d}{}_reco_unmatched_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight)) 

              hname = 'h{}_{}{:d}{}_gen_unmatched_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(t_pair_pt, np.log10(t_pair_weight)) 

              if jet_pt_det > 40:
                hname = 'h{}_{}{:d}{}_gen_unmatched_kin_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
                getattr(self, hname).Fill(t_pair_pt, np.log10(t_pair_weight)) 

              if not self.save_RUResponse:
                hname = 'THnF{}_{}{:d}{}_response_R{}_{}'.format(cone_label, observable, iRL, pair_type_label, jetR, obs_label)
                x = ([d_pair.pt, t_pair_pt, np.log10(d_pair.weight), np.log10(t_pair_weight)])
                x_array = array.array('d', x)
                getattr(self, hname).Fill(x_array)
      
  #---------------------------------------------------------------
  # Return EEC pairs with the input threshold cut
  # NB: this is not the most efficient implementation 
  # when using multiple threshold cuts 
  #---------------------------------------------------------------
  def get_EEC_pairs(self, jet, cone_parts, jet_pt, trk_thrd, ipoint=2, only_signal_pairs=True):

    pairs = []

    if cone_parts == None:
      constituents = fj.sorted_by_pt(jet.constituents())
    else:
      constituents = fj.sorted_by_pt(cone_parts)

    c_select = fj.vectorPJ()

    for c in constituents:
      if c.pt() < trk_thrd:
        break # NB: use the break statement since constituents are already sorted (so make sure the constituents are sorted)
      if only_signal_pairs:
        if c.user_index() >= 0:
          c_select.append(c) # NB: only consider 'signal' particles
      else:
        c_select.append(c)
    
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
      event_index1 = c_select[int(EEC_indicies1[i])].user_index()
      event_index2 = c_select[int(EEC_indicies2[i])].user_index()
      pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))

    return pairs

  #---------------------------------------------------------------
  # Return EEC pairs for cone particles
  #---------------------------------------------------------------
  def get_perpcone_EEC_pairs(self, cone_parts, jet_pt, trk_thrd, ipoint=2):

    pairs = []

    c_select = fj.vectorPJ()

    cone_parts_sorted = fj.sorted_by_pt(cone_parts) # make sure it's sorted before applying the threshold cur via "break"
    for c in cone_parts_sorted:
      if c.pt() < trk_thrd:
        break # NB: use the break statement since constituents are already sorted (so make sure the constituents are sorted)
      c_select.append(c)
    
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
      event_index1 = c_select[int(EEC_indicies1[i])].user_index()
      event_index2 = c_select[int(EEC_indicies2[i])].user_index()
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
