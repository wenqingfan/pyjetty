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

    # NB: match the strings in self.pair_type_label = ['bb','sb','ss']
    if type1 < 0 and type2 < 0:
      # print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp()
      return 0 # means bkg-bkg
    if type1 < 0 and type2 > 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 < 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 > 0:
      # print('sig-sig (',type1,type2,') pt1',constituents[part1].perp()
      return 2 # means sig-sig

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
      print('perp part with -999 index',self.index1) # double-check if the perp part index is -999

    if self.index2 < -999:
      type2 = 1 # jet part from embedding
    elif self.index2 >= 0:
      type2 = 1 # jet part from pythia
    else:
      type2 = -1 # perp part
      print('perp part with -999 index',self.index2) # double-check if the perp part index is -999

    # NB: match the strings in self.pair_type_label = ['bb','sb','ss']
    if type1 < 0 and type2 < 0:
      # print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp()
      return 0 # means bkg-bkg
    if type1 < 0 and type2 > 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 < 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 > 0 and type2 > 0:
      # print('sig-sig (',type1,type2,') pt1',constituents[part1].perp()
      return 2 # means sig-sig
  
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
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      binnings_truth = [np.linspace(-5,0,n_bins_truth[0]+1), \
                  np.array([5, 10, 20, 40, 60, 80, 100, 150]).astype(float) ]
      # slight difference for reco pt bin
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      n_bins_reco = [20, 6]
      binnings_reco = [np.linspace(-5,0,n_bins_reco[0]+1), \
                  np.array([10, 20, 40, 60, 80, 100, 150]).astype(float) ]
    else:
      # define binnings
      # these are the truth level binnings
      n_bins_truth = [20, 10] # WARNING RooUnfold seg faults if too many bins used
      # these are the truth level binnings
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      binnings_truth = [np.linspace(-5,0,n_bins_truth[0]+1), \
                  np.array([20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200]).astype(float) ]
      # slight difference for reco pt bin
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      n_bins_reco = [20, 6]
      binnings_reco = [np.linspace(-5,0,n_bins_reco[0]+1), \
                  np.array([40, 50, 60, 70, 80, 100, 120]).astype(float) ]

      self.n_RLbins = 30
      self.RLbins = logbins(1E-2,1,self.n_RLbins)

    self.pair_type_labels = ['']
    if self.do_rho_subtraction:
      self.pair_type_labels = ['_bb','_sb','_ss']

    perpcone_R_list = []
    if self.do_jetcone:
      if self.do_only_jetcone:
        for jetcone_R in self.jetcone_R_list:
          perpcone_R_list.append(jetcone_R)
      else:
        perpcone_R_list.append(jetR)
        for jetcone_R in self.jetcone_R_list:
          if jetcone_R != jetR: # just a safeguard since jetR is already added in the list
            perpcone_R_list.append(jetcone_R)
    else:
      perpcone_R_list.append(jetR)

    for observable in self.observable_list:
      
      # can take EEC with different energy power (currently only EEC with power n = 1 implemented)
      if observable == 'jet_ENC_RL':
      
        for trk_thrd in self.obs_settings[observable]:

          obs_label = self.utils.obs_label(trk_thrd, None) 

          #=======================================
          #        Sigma_ENC histogram
          #=======================================
          ######### only signal pairs ##########
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

          ######### all pair types ##########
          for pair_type_label in self.pair_type_labels:

            name = 'h_{}_sigma_{}_reco_unmatched_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetYaxis().SetTitle('R^{det}_{L}')
            h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
            setattr(self, name, h)

            name = 'h_{}_sigma_{}_gen_unmatched_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetYaxis().SetTitle('R^{det}_{L}')
            h.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
            setattr(self, name, h)

            if self.do_perpcone:
              for perpcone_R in perpcone_R_list:
                name = 'h_perpcone{}_{}_sigma_{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetYaxis().SetTitle('R^{det}_{L}')
                h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                setattr(self, name, h)

                name = 'h_perpcone{}_{}_sigma_{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, pair_type_label, jetR, obs_label)
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
          #=======================================
          for iRL in range(self.n_RLbins):
            
            ######### only signal pairs ##########
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

            ######### all pair types ##########
            for pair_type_label in self.pair_type_labels:

              name = 'h_{}{:d}_{}_reco_unmatched_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
              h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
              h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
              h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
              setattr(self, name, h2_reco)

              name = 'h_{}{:d}_{}_gen_unmatched_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
              h2_gen = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
              h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
              h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
              setattr(self, name, h2_gen)

              if self.do_perpcone:
                for perpcone_R in perpcone_R_list:
                  name = 'h_perpcone{}_{}{:d}_{}_reco_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_reco = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_reco.GetYaxis().SetTitle('log10(weight^{det})')
                  h2_reco.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
                  setattr(self, name, h2_reco)

                  name = 'h_perpcone{}_{}{:d}_{}_gen_unmatched_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                  h2_gen = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
                  h2_gen.GetYaxis().SetTitle('log10(weight^{truth})')
                  h2_gen.GetXaxis().SetTitle('p^{truth}_{T,ch jet}')
                  setattr(self, name, h2_gen)
            
            ######### only signal pairs ##########
            if self.save_RUResponse:
              # save response matrix in RooUnfoldResponse format directly
              # fill misses pair by pair
              name = 'h_{}{:d}_response_R{}_{}'.format(observable, iRL, jetR, obs_label)
              response = ROOT.RooUnfoldResponse(h2_reco, h2_gen)
              response.SetName(name)
              setattr(self, name, response)
            else:
              ######### only signal pairs ##########
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

              ######### all pair types ##########
              for pair_type_label in self.pair_type_labels:
                name = 'THnF_{}{:d}_{}_response_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
                self.create_thn(name, title, dim, nbins, min, max)

                if self.do_perpcone:
                  for perpcone_R in perpcone_R_list:
                    name = 'THnF_perpcone{}_{}{:d}_{}_response_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    self.create_thn(name, title, dim, nbins, min, max)

            # for purity correction
            name = 'h_{}{:d}_reco_unmatched_R{}_{}'.format(observable, iRL, jetR, obs_label)
            h = ROOT.TH2D(name, name, n_bins_reco[1], binnings_reco[1], n_bins_reco[0], binnings_reco[0])
            h.GetYaxis().SetTitle('log10(weight^{det})')
            h.GetXaxis().SetTitle('p^{det}_{T,ch jet}')
            setattr(self, name, h) 

          # RL resolution check for pairs
          name = 'h2d_matched_pair_RL_truth_vs_det_R{}_{}'.format(jetR, obs_label)
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
                                 grooming_setting, obs_label, jet_pt_ungroomed):
    return

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

    if self.do_rho_subtraction:
      jet_pt_det = jet_pt_det_ungroomed
    else:
      jet_pt_det = jet_det.perp()

    trk_thrd = obs_setting

    hname = 'h_jetpt_reco1D_matched_R{}_{}'.format(jetR, obs_label)
    getattr(self, hname).Fill(jet_pt_det)
    hname = 'h_jetpt_gen1D_matched_R{}_{}'.format(jetR, obs_label)
    getattr(self, hname).Fill(jet_truth.perp())
    
    if self.save_RUResponse:
      hname = 'h_jetpt_response1D_R{}_{}'.format(jetR, obs_label)
      getattr(self, hname).Fill(jet_pt_det, jet_truth.perp(), self.pt_hat)
    else:
      hname = 'THnF_jetpt_response1D_R{}_{}'.format(jetR, obs_label)
      getattr(self, hname).Fill(jet_pt_det, jet_truth.perp())

    for observable in self.observable_list:
      
      if observable == 'jet_ENC_RL':
        
        # type 1 -- fill for jet constituents
        if (cone_R == 0) and (cone_parts_in_det_jet == None):

          ################### all pair types ###################
          det_pairs_all = self.get_EEC_pairs(jet_det, jet_pt_det, trk_thrd, ipoint=2, only_signal_pairs=False)

          for d_pair in det_pairs_all:
            pair_type = d_pair.jet_pair_type() 
            pair_type_label = self.pair_type_labels[pair_type]
            print('pair index1', d_pair.index1, 'index2', d_pair.index2, 'pair type', pair_type, 'label', pair_type_label)

            hname = 'h_{}_sigma_{}_reco_unmatched_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

            t_pair_pt = jet_truth.pt()
            t_pair_weight = d_pair.weight*d_pair.pt*d_pair.pt/(t_pair_pt*t_pair_pt) # replace det jet pt by truth jet pt in the energy weight calculation

            hname = 'h_{}_sigma_{}_gen_unmatched_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(t_pair_pt, d_pair.r, t_pair_weight) # fill det RL (assuming very similar det RL and truth RL)

            # determine RL bin for det pairs
            iRL = bisect(self.RLbins, d_pair.r)-1 # index from 0

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h_{}{:d}_{}_reco_unmatched_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight)) 

              hname = 'h_{}{:d}_{}_gen_unmatched_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(t_pair_pt, np.log10(t_pair_weight)) 

              if not self.save_RUResponse:
                hname = 'THnF_{}{:d}_{}_response_R{}_{}'.format(observable, iRL, pair_type_label, jetR, obs_label)
                x = ([d_pair.pt, t_pair_pt, np.log10(d_pair.weight), np.log10(t_pair_weight)])
                x_array = array.array('d', x)
                getattr(self, hname).Fill(x_array)

          
          ################### Onlt signal pairs ###################
          # truth level EEC pairs
          truth_pairs = self.get_EEC_pairs(jet_truth, jet_truth.perp(), trk_thrd, ipoint=2, only_signal_pairs=True)

          # det level EEC pairs (only ss pairs)
          det_pairs = self.get_EEC_pairs(jet_det, jet_pt_det, trk_thrd, ipoint=2, only_signal_pairs=True)          

          ######### purity correction #########
          # calculate det EEC cross section irregardless if truth match exists

          for d_pair in det_pairs:

            hname = 'h_{}_sigma_reco_unmatched_R{}_{}'.format(observable, jetR, obs_label)
            getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

            # determine RL bin for det pairs
            iRL = bisect(self.RLbins, d_pair.r)-1 # index from 0

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h_{}{:d}_reco_unmatched_R{}_{}'.format(observable, iRL, jetR, obs_label)
              getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight))

          ########################## Fill RM (matched and missed) #########################
          dummyval = -9999

          # pair matching
          for t_pair in truth_pairs:

            hname = 'h_{}_sigma_gen_unmatched_R{}_{}'.format(observable, jetR, obs_label)
            getattr(self, hname).Fill(t_pair.pt, t_pair.r, t_pair.weight)

            if jet_pt_det > 40:
              hname = 'h_{}_sigma_gen_unmatched_kin_R{}_{}'.format(observable, jetR, obs_label)
              getattr(self, hname).Fill(t_pair.pt, t_pair.r, t_pair.weight)

            # determine RL bin fr truth pairs
            iRL = bisect(self.RLbins, t_pair.r)-1 # index from 0

            # if iRL ==40:
            #   print("new trurh pair")

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h_{}{:d}_gen_unmatched_R{}_{}'.format(observable, iRL, jetR, obs_label)
              getattr(self, hname).Fill(t_pair.pt, np.log10(t_pair.weight))

              # kinematic effect because of truncation in matched det pT
              if jet_pt_det > 40: # hard code the truncation value as 40 for now
                hname = 'h_{}{:d}_gen_unmatched_kin_R{}_{}'.format(observable, iRL, jetR, obs_label)
                getattr(self, hname).Fill(t_pair.pt, np.log10(t_pair.weight))
              # if iRL == 40:
              #   print('gen pair with distance',t_pair.r,'weight',t_pair.weight,'pt',t_pair.pt)

            match_found = False
            for d_pair in det_pairs:

              if d_pair.is_equal(t_pair):

                hname = 'h_{}_sigma_gen_matched_R{}_{}'.format(observable, jetR, obs_label)
                getattr(self, hname).Fill(t_pair.pt, t_pair.r, t_pair.weight)

                hname = 'h_{}_sigma_reco_matched_R{}_{}'.format(observable, jetR, obs_label)
                getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

                # fill the RL at det v.s. truth level (no energy weight)
                hname = 'h2d_matched_pair_RL_truth_vs_det_R{}_{}'.format(jetR, obs_label)
                getattr(self, hname).Fill(d_pair.r, t_pair.r)
                
                # if iRL == 40:
                #   print('matched reco pair with distance',d_pair.r,'weight',d_pair.weight,'pt',d_pair.pt)

                # NB: assuming very similar d_pair.r and t_pair.r
                if iRL >= 0 and iRL < self.n_RLbins:

                  hname = 'h_{}{:d}_gen_matched_R{}_{}'.format(observable, iRL, jetR, obs_label)
                  getattr(self, hname).Fill(t_pair.pt, np.log10(t_pair.weight))

                  hname = 'h_{}{:d}_reco_matched_R{}_{}'.format(observable, iRL, jetR, obs_label)
                  getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight))
                  
                  if self.save_RUResponse:
                    hname = 'h_{}{:d}_response_R{}_{}'.format(observable, iRL, jetR, obs_label, self.pt_hat) # NB: if RooUnfoldResponse format, applying scaling during while processing
                    getattr(self, hname).Fill(d_pair.pt, d_pair.weight, t_pair.pt, t_pair.weight)
                  else:
                    hname = 'THnF_{}{:d}_response_R{}_{}'.format(observable, iRL, jetR, obs_label)
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
                  hname = 'h_{}{:d}_response_R{}_{}'.format(observable, iRL, jetR, obs_label)
                  getattr(self, hname).Miss(t_pair.pt, t_pair.weight, self.pt_hat)  # NB: if RooUnfoldResponse format, applying scaling during while processing
                else:
                  hname = 'THnF_{}{:d}_response_miss_R{}_{}'.format(observable, iRL, jetR, obs_label)
                  x = ([t_pair.pt, np.log10(t_pair.weight), jet_pt_det])
                  x_array = array.array('d', x)
                  getattr(self, hname).Fill(x_array)

        # type 2 -- fill for perp cone
        if (self.do_perpcone) and (cone_R > 0) and (cone_parts_in_det_jet != None) and (cone_parts_in_truth_jet == None): 

          ################### all pair types ###################
          det_pairs_all = self.get_perpcone_EEC_pairs(cone_parts_in_det_jet, jet_pt_det, trk_thrd, ipoint=2) 

          # FIX ME: what about the purity effect?
          for d_pair in det_pairs_all:
            pair_type = d_pair.perpcone_pair_type() 
            pair_type_label = self.pair_type_labels[pair_type]

            hname = 'h_perpcone{}_{}_sigma_{}_reco_unmatched_R{}_{}'.format(cone_R, observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(d_pair.pt, d_pair.r, d_pair.weight)

            t_pair_pt = jet_truth.pt()
            t_pair_weight = d_pair.weight*d_pair.pt*d_pair.pt/(t_pair_pt*t_pair_pt) # replace det jet pt by truth jet pt in the energy weight calculation

            hname = 'h_perpcone{}_{}_sigma_{}_gen_unmatched_R{}_{}'.format(cone_R, observable, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(t_pair_pt, d_pair.r, t_pair_weight) # fill det RL (assuming very similar det RL and truth RL)

            # determine RL bin for det pairs
            iRL = bisect(self.RLbins, d_pair.r)-1 # index from 0

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h_perpcone{}_{}{:d}_{}_reco_unmatched_R{}_{}'.format(cone_R, observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(d_pair.pt, np.log10(d_pair.weight)) 

              hname = 'h_perpcone{}_{}{:d}_{}_gen_unmatched_R{}_{}'.format(cone_R, observable, iRL, pair_type_label, jetR, obs_label)
              getattr(self, hname).Fill(t_pair_pt, np.log10(t_pair_weight)) 

              if not self.save_RUResponse:
                hname = 'THnF_perpcone{}_{}{:d}_{}_response_R{}_{}'.format(cone_R, observable, iRL, pair_type_label, jetR, obs_label)
                x = ([d_pair.pt, t_pair_pt, np.log10(d_pair.weight), np.log10(t_pair_weight)])
                x_array = array.array('d', x)
                getattr(self, hname).Fill(x_array)
      
  #---------------------------------------------------------------
  # Return EEC pairs with the input threshold cut
  # NB: this is not the most efficient implementation 
  # when using multiple threshold cuts 
  #---------------------------------------------------------------
  def get_EEC_pairs(self, jet, jet_pt, trk_thrd, ipoint=2, only_signal_pairs=True):

    pairs = []

    constituents = fj.sorted_by_pt(jet.constituents())
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

    print('double check: 1st cone parts size is',len(cone_parts))
    for c in cone_parts:
      print('double check: part with index',c.user_index())

    for c in cone_parts:
      # if c.pt() < trk_thrd:
      #   break # NB: use the break statement since constituents are already sorted (so make sure the constituents are sorted)
      c_select.append(c)
      if c.user_index()==-999:
        print('double check: found a perp part with index',c.user_index())
    
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
