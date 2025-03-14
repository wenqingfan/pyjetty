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
import random
from bisect import bisect

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import ecorrel

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

    if self.is_pp:
      # define binnings
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      n_bins = [20, 30]
      binnings = [np.linspace(-5,0,n_bins[0]+1), \
                  np.linspace(0,150,n_bins[1]+1) ]

      self.n_RLbins = 25
      self.RLbins = logbins(1E-2,1,self.n_RLbins)
    else:
      # define binnings
      # binnings[0] -- log10(weight), binnings[1] -- jet pT
      n_bins = [20, 30]
      binnings = [np.linspace(-5,0,n_bins[0]+1), \
                  np.linspace(0,150,n_bins[1]+1) ]

      self.n_RLbins = 25
      self.RLbins = logbins(1E-2,1,self.n_RLbins)

    for jetR in self.jetR_list:
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
        for trk_thrd in self.obs_settings[observable]:

          obs_label = self.utils.obs_label(trk_thrd, None) 
          
          # can take EEC with different energy power (currently only EEC with power n = 1 implemented)
          if 'jet_ENC_RL' in observable:

            #=======================================
            #        Sigma_ENC histogram
            #=======================================
            name = 'h_{}_sigma_R{}_{}'.format(observable, jetR, obs_label)
            pt_bins = linbins(0,200,40)
            h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
            h.GetXaxis().SetTitle('p_{T,ch jet}')
            h.GetYaxis().SetTitle('R_{L}')
            setattr(self, name, h)

            #=======================================
            #      1D unfolding for jet pT
            #=======================================
            # 1D to 1D RM
            name = 'h_jetpt1D_R{}_{}'.format(jetR, obs_label)
            h = ROOT.TH1D(name, name, n_bins[1], binnings[1])
            h.GetXaxis().SetTitle('p_{T,ch jet}')
            h.GetYaxis().SetTitle('Counts')
            setattr(self, name, h)
            
            for iRL in range(self.n_RLbins):

              #=======================================
              # 2D unfolding for energy correlators
              #=======================================
              name = 'h_{}{:d}_R{}_{}'.format(observable, iRL, jetR, obs_label)
              h = ROOT.TH2D(name, name, n_bins[1], binnings[1], n_bins[0], binnings[0])
              h.GetYaxis().SetTitle('log10(weight)')
              h.GetXaxis().SetTitle('p_{T,ch jet}')
              setattr(self, name, h)

          # fill perp cone histograms
          self.pair_type_labels = ['']

          if self.do_rho_subtraction:
            self.pair_type_labels = ['_bb','_sb','_ss']
          
          if self.do_perpcone:
            
            for perpcone_R in perpcone_R_list:

              for pair_type_label in self.pair_type_labels:

                if 'jet_ENC_RL' in observable:
                  
                  #=======================================
                  #        Sigma_ENC histogram
                  #=======================================
                  name = 'h_perpcone{}_{}_sigma_R{}_{}'.format(perpcone_R, observable + pair_type_label, jetR, obs_label)
                  pt_bins = linbins(0,200,40)
                  h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                  h.GetXaxis().SetTitle('p_{T,ch jet}')
                  h.GetYaxis().SetTitle('R_{L}')
                  setattr(self, name, h)
                  
                  for iRL in range(self.n_RLbins):

                    #=======================================
                    # 2D unfolding for energy correlators
                    #=======================================
                    name = 'h_perpcone{}_{}{:d}{}_R{}_{}'.format(perpcone_R, observable, iRL, pair_type_label, jetR, obs_label)
                    h = ROOT.TH2D(name, name, n_bins[1], binnings[1], n_bins[0], binnings[0])
                    h.GetYaxis().SetTitle('log10(weight)')
                    h.GetXaxis().SetTitle('p_{T,ch jet}')
                    setattr(self, name, h)

          if self.do_jetcone:

            for jetcone_R in self.jetcone_R_list:
              
              if 'jet_ENC_RL' in observable:
                
                #=======================================
                #        Sigma_ENC histogram
                #=======================================
                name = 'h_jetcone{}_{}_sigma_R{}_{}'.format(jetcone_R, observable, jetR, obs_label)
                pt_bins = linbins(0,200,40)
                h = ROOT.TH2D(name, name, 40, pt_bins, self.n_RLbins, self.RLbins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)
                
                for iRL in range(self.n_RLbins):

                  #=======================================
                  # 2D unfolding for energy correlators
                  #=======================================
                  name = 'h_jetcone{}_{}{:d}_R{}_{}'.format(jetcone_R, observable, iRL, jetR, obs_label)
                  h = ROOT.TH2D(name, name, n_bins[1], binnings[1], n_bins[0], binnings[0])
                  h.GetYaxis().SetTitle('log10(weight)')
                  h.GetXaxis().SetTitle('p_{T,ch jet}')
                  setattr(self, name, h)
                    
  #---------------------------------------------------------------
  # Calculate pair distance of two fastjet particles
  #---------------------------------------------------------------
  def calculate_distance(self, p0, p1):   
    dphiabs = math.fabs(p0.phi() - p1.phi())
    dphi = dphiabs

    if dphiabs > math.pi:
      dphi = 2*math.pi - dphiabs

    deta = p0.eta() - p1.eta()
    return math.sqrt(deta*deta + dphi*dphi)

  def is_same_charge(self, corr_builder, ipoint, constituents, index):
    part1 = int(corr_builder.correlator(ipoint).indices1()[index])
    part2 = int(corr_builder.correlator(ipoint).indices2()[index])
    q1 = int(constituents[part1].python_info().charge)
    q2 = int(constituents[part2].python_info().charge)

    if q1*q2 > 0:
      return True
    else:
      return False

  def check_pair_type(self, corr_builder, ipoint, constituents, index):
    part1 = int(corr_builder.correlator(ipoint).indices1()[index])
    part2 = int(corr_builder.correlator(ipoint).indices2()[index])
    type1 = constituents[part1].user_index()
    type2 = constituents[part2].user_index()

    # NB: match the strings in self.pair_type_label = ['bb','sb','ss']
    if type1 < 0 and type2 < 0:
      # print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp()
      return 0 # means bkg-bkg
    if type1 < 0 and type2 >= 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 >= 0 and type2 < 0:
      # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
      return 1 # means sig-bkg
    if type1 >= 0 and type2 >= 0:
      # print('sig-sig (',type1,type2,') pt1',constituents[part1].perp()
      return 2 # means sig-sig

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix):

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()
    trk_thrd = obs_setting

    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted
    nconst_jet = len(c_select)

    if self.ENC_pair_cut:
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      if jet.area() == 0:
        return # NB: skip the zero area jets for now (also skip the perp-cone and jet-cone w.r.t. the zero area jets)
    else:
      jet_pt = jet.perp()
    # print('unsubtracted pt',jet.perp(),'subtracted',jet_pt,'# of constituents >',trk_thrd,'is',len(c_select))

    for observable in self.observable_list:

      if observable == 'jet_ENC_RL':

        hname = 'h_jetpt1D_R{}_{}'.format(jetR, obs_label)
        getattr(self, hname).Fill(jet_pt)
        
        # if analyze jet cones and only analyze jet cones, then only fill jet pt histograms for standard jets (just to speed things up)
        if self.do_jetcone and self.do_only_jetcone:
          pass
        else:
          new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, 2, 1, dphi_cut, deta_cut)

          ipoint = 2
          for index in range(new_corr.correlator(ipoint).rs().size()):

            # processing only like-sign pairs when self.ENC_pair_like is on
            if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
              continue

            # processing only unlike-sign pairs when self.ENC_pair_unlike is on
            if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
              continue

            RL = new_corr.correlator(ipoint).rs()[index]
            weight = new_corr.correlator(ipoint).weights()[index]
            
            hname = 'h_{}_sigma_R{}_{}'.format(observable, jetR, obs_label)
            getattr(self, hname).Fill(jet_pt, RL, weight)

            # determine RL bin for det pairs
            iRL = bisect(self.RLbins, RL)-1 # index from 0

            if iRL >= 0 and iRL < self.n_RLbins:
              hname = 'h_{}{:d}_R{}_{}'.format(observable, iRL, jetR, obs_label)
              getattr(self, hname).Fill(jet_pt, np.log10(weight))
          
  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_perp_cone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):

    # combine sig jet and perp cone with trk threshold cut
    trk_thrd = obs_setting
    c_select = fj.vectorPJ()

    cone_parts_sorted = fj.sorted_by_pt(cone_parts)
    # print('perp cone nconst:',len(cone_parts_sorted))
    for part in cone_parts_sorted:
      if part.pt() < trk_thrd:
        break
      c_select.append(part) # NB: use the break statement since constituents are already sorted

    if self.ENC_pair_cut:
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      if jet.area() == 0:
        return # NB: skip the zero area jets for now (also skip the perp-cone and jet-cone w.r.t. the zero area jets)
    else:
      jet_pt = jet.perp()
    
    for observable in self.observable_list:

      if observable == 'jet_ENC_RL':

        new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, 2, 1, dphi_cut, deta_cut)

        ipoint = 2
        for index in range(new_corr.correlator(ipoint).rs().size()):

          # processing only like-sign pairs when self.ENC_pair_like is on
          if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
            continue

          # processing only unlike-sign pairs when self.ENC_pair_unlike is on
          if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
            continue

          # separate out sig-sig, sig-bkg, bkg-bkg correlations for EEC pairs
          pair_type_label = ''
          if self.do_rho_subtraction:
            pair_type = self.check_pair_type(new_corr, ipoint, c_select, index)
            pair_type_label = self.pair_type_labels[pair_type]

          RL = new_corr.correlator(ipoint).rs()[index]
          weight = new_corr.correlator(ipoint).weights()[index]
          
          hname = 'h_perpcone{}_{}_sigma_R{}_{}'.format(cone_R, observable + pair_type_label, jetR, obs_label)
          getattr(self, hname).Fill(jet_pt, RL, weight)

          # determine RL bin for det pairs
          iRL = bisect(self.RLbins, RL)-1 # index from 0

          if iRL >= 0 and iRL < self.n_RLbins:
            hname = 'h_perpcone{}_{}{:d}{}_R{}_{}'.format(cone_R, observable, iRL, pair_type_label, jetR, obs_label)
            getattr(self, hname).Fill(jet_pt, np.log10(weight))

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_jet_cone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):

    # combine sig jet and perp cone with trk threshold cut
    trk_thrd = obs_setting
    c_select = fj.vectorPJ()

    cone_parts_sorted = fj.sorted_by_pt(cone_parts)
    # print('perp cone nconst:',len(cone_parts_sorted))
    for part in cone_parts_sorted:
      if part.pt() < trk_thrd:
        break
      c_select.append(part) # NB: use the break statement since constituents are already sorted

    if self.ENC_pair_cut:
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    if self.do_rho_subtraction:
      jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      if jet.area() == 0:
        return # NB: skip the zero area jets for now (also skip the perp-cone and jet-cone w.r.t. the zero area jets)
    else:
      jet_pt = jet.perp()
    
    for observable in self.observable_list:

      if observable == 'jet_ENC_RL':

        new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, 2, 1, dphi_cut, deta_cut)

        ipoint = 2
        for index in range(new_corr.correlator(ipoint).rs().size()):

          # processing only like-sign pairs when self.ENC_pair_like is on
          if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
            continue

          # processing only unlike-sign pairs when self.ENC_pair_unlike is on
          if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
            continue

          RL = new_corr.correlator(ipoint).rs()[index]
          weight = new_corr.correlator(ipoint).weights()[index]
          
          hname = 'h_jetcone{}_{}_sigma_R{}_{}'.format(cone_R, observable, jetR, obs_label)
          getattr(self, hname).Fill(jet_pt, RL, weight)

          # determine RL bin for det pairs
          iRL = bisect(self.RLbins, RL)-1 # index from 0

          if iRL >= 0 and iRL < self.n_RLbins:
            hname = 'h_jetcone{}_{}{:d}_R{}_{}'.format(cone_R, observable, iRL, jetR, obs_label)
            getattr(self, hname).Fill(jet_pt, np.log10(weight))

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
