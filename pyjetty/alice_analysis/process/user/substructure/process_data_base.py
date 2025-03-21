#!/usr/bin/env python3

"""
Base class to read a ROOT TTree of track information
and do jet-finding, and save basic histograms.
  
To use this class, the following should be done:

  - Implement a user analysis class inheriting from this one, such as in user/james/process_data_XX.py
    You should implement the following functions:
      - initialize_user_output_objects()
      - fill_jet_histograms()
    
  - The histogram of the data should be named h_[obs]_JetPt_R[R]_[subobs]_[grooming setting]
    The grooming part is optional, and should be labeled e.g. zcut01_B0 — from CommonUtils::grooming_label({'sd':[zcut, beta]})
    For example: h_subjet_z_JetPt_R0.4_0.1
    For example: h_subjet_z_JetPt_R0.4_0.1_zcut01_B0

  - You also should modify observable-specific functions at the top of common_utils.py
  
Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import time

# Data analysis and plotting
import numpy as np
import os
import ROOT
import yaml
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_base
from pyjetty.mputils.csubtractor import CEventSubtractor

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class ProcessDataBase(process_base.ProcessBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessDataBase, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

    # Initialize configuration
    self.initialize_config()
    
  #---------------------------------------------------------------
  # Initialize config file into class members
  #---------------------------------------------------------------
  def initialize_config(self):
    
    # Call base class initialization
    process_base.ProcessBase.initialize_config(self)
    
    # Read config file
    with open(self.config_file, 'r') as stream:
      config = yaml.safe_load(stream)
    
    if self.do_constituent_subtraction:
      self.is_pp = False
    else:
      self.is_pp = True

    if 'ENC_pair_cut' in config:
        self.ENC_pair_cut = config['ENC_pair_cut']
    else:
        self.ENC_pair_cut = False
    if 'ENC_pair_like' in config:
        self.ENC_pair_like = config['ENC_pair_like']
    else:
        self.ENC_pair_like = False
    if 'ENC_pair_unlike' in config:
        self.ENC_pair_unlike = config['ENC_pair_unlike']
    else:
        self.ENC_pair_unlike = False

    if 'do_rho_subtraction' in config:
      self.do_rho_subtraction = config['do_rho_subtraction']
    else:
      self.do_rho_subtraction = False

    if 'do_perpcone' in config:
      self.do_perpcone = config['do_perpcone']
    else:
      self.do_perpcone = False
    if 'static_perpcone' in config:
        self.static_perpcone = config['static_perpcone']
    else:
        self.static_perpcone = True # NB: set default to rigid cone (less fluctuations)
    if 'do_randomcone' in config:
        self.do_randomcone = config['do_randomcone']
        self.do_perpcone = self.do_randomcone
    else:
        self.do_randomcone = False
    if self.do_randomcone:
        seed = (int(time.time() * 1000) + os.getpid()) % (2**32)
        np.random.seed(seed)

    if 'do_jetcone' in config:
      self.do_jetcone = config['do_jetcone']
    else:
      self.do_jetcone = False
    if self.do_jetcone and 'jetcone_R_list' in config:
      self.jetcone_R_list = config['jetcone_R_list']
    else:
      self.jetcone_R_list = [0.4] # NB: set default value to 0.4
    if 'do_only_jetcone' in config:
      self.do_only_jetcone = config['do_only_jetcone']
    else:
      self.do_only_jetcone = False

    if 'do_2cones' in config:
      self.do_2cones = config['do_2cones']
    else:
      self.do_2cones = False

    if 'leading_pt' in config:
        self.leading_pt = config['leading_pt']
    else:
        self.leading_pt = -1 # negative means no leading track cut

    # if only processing dijets 
    if 'leading_jet' in config:
      self.leading_jet = config['leading_jet']
    else:
      self.leading_jet = False
    if 'subleading_jet' in config:
      self.subleading_jet = config['subleading_jet']
    else:
      self.subleading_jet = False

    # NB: safeguard, make sure to only process one type at a time
    if (self.leading_jet and (not self.subleading_jet)) or ((not self.leading_jet) and self.subleading_jet):
      self.is_dijet = True
      self.xj_interval = 0.2
    else:
      self.is_dijet = False
    
    # Create dictionaries to store grooming settings and observable settings for each observable
    # Each dictionary entry stores a list of subconfiguration parameters
    #   The observable list stores the observable setting, e.g. subjetR
    #   The grooming list stores a list of grooming settings {'sd': [zcut, beta]} or {'dg': [a]}
    self.observable_list = config['process_observables']
    self.obs_settings = {}
    self.obs_grooming_settings = {}
    for observable in self.observable_list:
    
      obs_config_dict = config[observable]
      obs_config_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
      
      obs_subconfig_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
      self.obs_settings[observable] = self.utils.obs_settings(observable, obs_config_dict, obs_subconfig_list)
      self.obs_grooming_settings[observable] = self.utils.grooming_settings(obs_config_dict)
      
    # Construct set of unique grooming settings
    self.grooming_settings = []
    lists_grooming = [self.obs_grooming_settings[obs] for obs in self.observable_list]
    for observable in lists_grooming:
      for setting in observable:
        if setting not in self.grooming_settings and setting != None:
          self.grooming_settings.append(setting)
          
  #---------------------------------------------------------------
  # Main processing function
  #---------------------------------------------------------------
  def process_data(self):
    
    self.start_time = time.time()

    # Use IO helper class to convert ROOT TTree into a SeriesGroupBy object of fastjet particles per event
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    io = process_io.ProcessIO(input_file=self.input_file, track_tree_name='tree_Particle',
                              is_pp=self.is_pp, use_ev_id_ext=True)
    self.df_fjparticles = io.load_data(m=self.m)
    self.nEvents = len(self.df_fjparticles.index)
    self.nTracks = len(io.track_df.index)
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # Initialize histograms
    self.initialize_output_objects()
    
    # Create constituent subtractor, if configured
    if not self.is_pp:
      self.constituent_subtractor = [CEventSubtractor(max_distance=R_max, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR) for R_max in self.max_distance]
    
    print(self)

    # Find jets and fill histograms
    print('Analyze events...')
    self.analyze_events()
    
    # Plot histograms
    print('Save histograms...')
    process_base.ProcessBase.save_output_objects(self)

    print('--- {} seconds ---'.format(time.time() - self.start_time))
  
  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_output_objects(self):
  
    # Initialize user-specific histograms
    self.initialize_user_output_objects()
    
    # Initialize base histograms
    self.hNevents = ROOT.TH1F('hNevents', 'hNevents', 2, -0.5, 1.5)
    if self.event_number_max < self.nEvents:
      self.hNevents.Fill(1, self.event_number_max)
    else:
      self.hNevents.Fill(1, self.nEvents)
    
    self.hTrackEtaPhi = ROOT.TH2F('hTrackEtaPhi', 'hTrackEtaPhi', 200, -1., 1., 628, 0., 6.28)
    self.hTrackPt = ROOT.TH1F('hTrackPt', 'hTrackPt', 300, 0., 300.)
    
    if not self.is_pp:
      self.hRho = ROOT.TH1F('hRho', 'hRho', 1000, 0., 1000.)
        
    for jetR in self.jetR_list:
      
      name = 'hZ_R{}'.format(jetR)
      h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 1.)
      setattr(self, name, h)

  #---------------------------------------------------------------
  # Main function to loop through and analyze events
  #---------------------------------------------------------------
  def analyze_events(self):
    
    # Fill track histograms
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    print('Fill track histograms')
    [[self.fillTrackHistograms(track) for track in fj_particles] for fj_particles in self.df_fjparticles]
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    print('Find jets...')
    fj.ClusterSequence.print_banner()
    print()
    self.event_number = 0
  
    # Use list comprehension to do jet-finding and fill histograms
    result = [self.analyze_event(fj_particles) for fj_particles in self.df_fjparticles]
    
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    print('Save thn...')
    process_base.ProcessBase.save_thn_th3_objects(self)
  
  #---------------------------------------------------------------
  # Fill track histograms.
  #---------------------------------------------------------------
  def fillTrackHistograms(self, track):
    
    self.hTrackEtaPhi.Fill(track.eta(), track.phi())
    self.hTrackPt.Fill(track.pt())
  
  #---------------------------------------------------------------
  # Analyze jets of a given event.
  # fj_particles is the list of fastjet pseudojets for a single fixed event.
  #---------------------------------------------------------------
  def analyze_event(self, fj_particles):
  
    self.event_number += 1
    if self.event_number > self.event_number_max:
      return
    if self.debug_level > 1:
      print('-------------------------------------------------')
      print('event {}'.format(self.event_number))
    
    if len(fj_particles) > 1:
      if np.abs(fj_particles[0].pt() - fj_particles[1].pt()) <  1e-10:
        print('WARNING: Duplicate particles may be present')
        print([p.user_index() for p in fj_particles])
        print([p.pt() for p in fj_particles])
  
    # Perform constituent subtraction for each R_max (do this once, for all jetR)
    if not self.is_pp:
      fj_particles_subtracted = [self.constituent_subtractor[i].process_event(fj_particles) for i, R_max in enumerate(self.max_distance)]
    
    # Loop through jetR, and process event for each R
    for jetR in self.jetR_list:
    
      # Keep track of whether to fill R-independent histograms
      self.fill_R_indep_hists = (jetR == self.jetR_list[0])

      # Set jet definition and a jet selector
      jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
      jet_selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR)
      if self.debug_level > 2:
        print('jet definition is:', jet_def)
        print('jet selector is:', jet_selector,'\n')
        
      # Analyze
      if self.is_pp:
      
        # Do jet finding
        cs = fj.ClusterSequence(fj_particles, jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = jet_selector(jets)
      
        self.analyze_jets(jets_selected, jetR)
        
      else:
      
        for i, R_max in enumerate(self.max_distance):
                    
          if self.debug_level > 1:
            print('R_max: {}'.format(R_max))
            
          # Keep track of whether to fill R_max-independent histograms
          self.fill_Rmax_indep_hists = (i == 0)
          
          # Perform constituent subtraction
          rho = self.constituent_subtractor[i].bge_rho.rho()
          if self.fill_R_indep_hists and self.fill_Rmax_indep_hists:
            getattr(self, 'hRho').Fill(rho)
          
          # Do jet finding (re-do each time, to make sure matching info gets reset)
          cs = fj.ClusterSequence(fj_particles_subtracted[i], jet_def) # FIX ME: not sure whether to enable the area or not
          jets = fj.sorted_by_pt(cs.inclusive_jets())
          jets_selected = jet_selector(jets)

          # cs_unsub = fj.ClusterSequence(fj_particles, jet_def)
          cs_unsub = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
          jets_unsub = fj.sorted_by_pt(cs_unsub.inclusive_jets())
          jets_selected_unsub = jet_selector(jets_unsub)

          # debug
          # for jet in jets_selected_unsub:
          #   if jet.perp()-jet.area()*rho > 20:
          #     print('unsubtracted: jet pt',jet.perp(),'(',jet.perp()-jet.area()*rho,') eta',jet.eta(),'phi',jet.phi(),'area',jet.area(),'rho',rho,'product',jet.area()*rho)
          #     constituents = fj.sorted_by_pt(jet.constituents())
          #     if len(constituents)>0:
          #       print('leading part pt',constituents[0].perp(),'eta',constituents[0].eta(),'phi',constituents[0].phi())

          # for jet in jets_selected:
          #   if jet.perp() > 20:
          #     print('subtracted: jet pt',jet.perp(),'eta',jet.eta(),'phi',jet.phi())
          #     constituents = fj.sorted_by_pt(jet.constituents())
          #     if len(constituents)>0:
          #       print('leading part pt',constituents[0].perp(),'eta',constituents[0].eta(),'phi',constituents[0].phi())
          
          if self.do_rho_subtraction:
            if not (self.do_jetcone or self.do_perpcone):
              self.analyze_jets(jets_selected_unsub, jetR, R_max = R_max, rho_bge = rho)
            else:
              self.analyze_jets(jets_selected_unsub, jetR, R_max = R_max, rho_bge = rho, parts = fj_particles)
          else:
            if not (self.do_jetcone or self.do_perpcone):
              self.analyze_jets(jets_selected, jetR, R_max = R_max)
            else:
              self.analyze_jets(jets_selected, jetR, R_max = R_max, rho_bge = 0, parts = fj_particles) # NB: feed all particles for cone around the CS subtracted jet. An alternate way is to use CS subtracted particles

  #---------------------------------------------------------------
  # Jet selection cuts.
  #---------------------------------------------------------------
  def reselect_jets(self, jets_selected, jetR, rho_bge = 0):
    # re-apply jet pt > 5GeV cut after rho subtraction and leading track pt cut if there is any
    # no special treatment on jets when looking at perpcones (used to check perpcones for jets without leading pt cut)
    jets_reselected = fj.vectorPJ()
    for jet in jets_selected:
      is_jet_selected = True
      
      # leading track selection
      if self.leading_pt > 0:
        constituents = fj.sorted_by_pt(jet.constituents())
        if constituents[0].perp() < self.leading_pt:
          is_jet_selected = False
      
      # if rho subtraction, require jet pt > 5 after subtration
      if self.do_rho_subtraction and rho_bge > 0:
        if jet.perp()-rho_bge*jet.area() < 5:
          # FIX ME: not sure whether to apply the area selection or not yet. jet.area() > 0.6*np.pi*jetR*jetR
          is_jet_selected = False

      if is_jet_selected:
        jets_reselected.append(jet)

    return jets_reselected

  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets_selected, jetR, R_max = None, rho_bge = 0, parts = None):
  
    # Prepare suffix for the CS background subtraction config
    if R_max and (not self.do_rho_subtraction):
      R_max_label = '_Rmax{}'.format(R_max) # only use this suffix for "real" CS subtraction, not just rho subtraction
    else:
      R_max_label = ''

    # reselect jets after background subtraction (for PbPb case)
    jets_reselected = self.reselect_jets(jets_selected, jetR, rho_bge = rho_bge)

    # Loop through jets and call user function to fill histos
    if self.is_dijet:
      
      # in case background subtruction chnaged the ordering
      jets_reselected_sorted = fj.sorted_by_pt(jets_reselected)
      dijets = jets_reselected_sorted[:2]

      # accept events with # of jets >=2
      if len(dijets) < 2:
        return

      dphi = dijets[0].delta_phi_to(dijets[1])
      xj = dijets[1].perp()/dijets[0].perp()

      # back-to-back requirement
      if abs(dphi) < 5/6*math.pi:
        return
      
      # minimum pT cut on subleading jet
      if dijets[1].perp() < 10:
        return
      
      # print('dijet xj',xj,'(',dijets[1].perp(),'/',dijets[0].perp(),') dphi',dphi)

      ixjbin = int(xj/self.xj_interval)
      dijet_xj_label = '_xj{:.1f}{:.1f}'.format(0.2*ixjbin, 0.2*(ixjbin+1))

      # Set suffix for filling histograms
      suffix = '{}{}'.format(R_max_label, dijet_xj_label)

      if self.leading_jet: # leading
        result = self.analyze_accepted_jet(dijets[0], jetR, suffix, rho_bge)
      else: # subleading
        result = self.analyze_accepted_jet(dijets[1], jetR, suffix, rho_bge)
    
    else:

      # Set suffix for filling histograms
      suffix = '{}'.format(R_max_label)

      result = [self.analyze_accepted_jet(jet, jetR, suffix, rho_bge, parts) for jet in jets_reselected]

  #---------------------------------------------------------------
  # Fill histograms
  #---------------------------------------------------------------
  def analyze_accepted_jet(self, jet, jetR, suffix, rho_bge = 0, parts = None):
    
    # Check additional acceptance criteria
    if not self.utils.is_det_jet_accepted(jet):
      return
          
    # when using rho subtraction for PbPb analysis, skip jets with no area info or with zero area (NB: need to check how often this happens and if this results in any bias)
    if self.do_rho_subtraction:
      if jet.has_area():
        if jet.area() == 0:
          return
      else:
        return

    # Fill base histograms
    if self.do_rho_subtraction and rho_bge > 0:
      jet_pt_ungroomed = jet.pt() - rho_bge*jet.area()
    else:
      jet_pt_ungroomed = jet.pt()

    if self.is_pp or self.fill_Rmax_indep_hists:
    
      hZ = getattr(self, 'hZ_R{}'.format(jetR))
      for constituent in jet.constituents():
        z = constituent.pt() / jet_pt_ungroomed
        hZ.Fill(jet_pt_ungroomed, z)
    
    # Loop through each jet subconfiguration (i.e. subobservable / grooming setting)
    # Note that the subconfigurations are defined by the first observable, if multiple are defined
    observable = self.observable_list[0]
    for i in range(len(self.obs_settings[observable])):
    
      obs_setting = self.obs_settings[observable][i]
      grooming_setting = self.obs_grooming_settings[observable][i]
      obs_label = self.utils.obs_label(obs_setting, grooming_setting)
    
      # Groom jet, if applicable
      if grooming_setting:
        gshop = fjcontrib.GroomerShop(jet, jetR, self.reclustering_algorithm)
        jet_groomed_lund = self.utils.groom(gshop, grooming_setting, jetR)
        if not jet_groomed_lund:
          continue
      else:
        jet_groomed_lund = None

      # Call user function to fill histograms
      self.fill_jet_histograms(jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                               obs_label, jet_pt_ungroomed, suffix)

      # Fill histograms for jetcone
      if self.do_jetcone:
        for jetcone_R in self.jetcone_R_list:
          parts_in_cone = self.find_parts_around_jet(parts, jet, jetcone_R)
          self.fill_jet_cone_histograms(parts_in_cone, jetcone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                               obs_label, jet_pt_ungroomed, suffix, rho_bge)

      # Control rotation angle = rotation_sign*pi/2
      rotation_sign = 1.
      if self.do_randomcone:
          rotation_sign = np.random.uniform(4./3., 2./3.)
      
      # Fill histograms for perpcone
      if self.do_perpcone and not self.do_2cones:
        
        # construct perpcone size list
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

        # construct perp cones and fill histograms
        for perpcone_R in perpcone_R_list:

          parts_in_cone1 = self.construct_parts_in_perpcone(parts, jet, jetR, perpcone_R, +rotation_sign)
          self.fill_perp_cone_histograms(parts_in_cone1, perpcone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                               obs_label, jet_pt_ungroomed, suffix, rho_bge)

          parts_in_cone2 = self.construct_parts_in_perpcone(parts, jet, jetR, perpcone_R, -rotation_sign)
          self.fill_perp_cone_histograms(parts_in_cone2, perpcone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                               obs_label, jet_pt_ungroomed, suffix, rho_bge)

      # Fill histograms for two perpcone (currently cannot run both 1 cone and 2 cones for perpcone)
      if self.do_perpcone and self.do_2cones:
        
        # construct perpcone size list
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

        # construct perp cones and fill histograms
        for perpcone_R in perpcone_R_list:

          parts_in_cone = self.construct_parts_in_2perpcone(parts, jet, jetR, perpcone_R, rotation_sign)
          self.fill_perp_cone_histograms(parts_in_cone, perpcone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix, rho_bge)

  def construct_parts_in_perpcone(self, parts, jet, jetR, perpcone_R, rotation_sign):

    # print('jet pt',jet.perp()-rho_bge*jet.area(),'phi',jet.phi(),'eta',jet.eta(),'area',jet.area())

    perp_jet = fj.PseudoJet()
    perp_jet.reset_PtYPhiM(jet.pt(), jet.rapidity(), jet.phi() + rotation_sign*np.pi/2, jet.m())

    perpcone_R_effective = perpcone_R
    # Use jet cone parts as "signal" for perp cone if cone radius != jetR or if we are only checking the jetcones (in this case, use cone particles for jetcone R = jetR also), else use jet constituents as "signal" for perp cone
    if self.do_only_jetcone or perpcone_R != jetR:
      parts_in_jet = self.find_parts_around_jet(parts, jet, perpcone_R)
    else:
      constituents = jet.constituents()
      parts_in_jet = self.copy_parts(constituents) # NB: make a copy so that the original jet constituents will not be modifed
      if self.do_rho_subtraction and self.static_perpcone == False:
        perpcone_R_effective = math.sqrt(jet.area()/np.pi) # NB: for dynamic cone size

    # NB1: a deep copy already created in find_parts_around_jet(). Operations on the deep copy does not affect the oringinal parts     
    # NB2: when iterating using for loop through all the particle list like "for part in parts", operations like part.* will not change the parts. Need to use parts[*].* to change the parts  
    parts_in_perpcone = self.find_parts_around_jet(parts, perp_jet, perpcone_R_effective)
    # for part in parts_in_perpcone:
    #   print('before rotation (pt, eta, phi)',part.pt(),part.eta(),part.phi())
    parts_in_perpcone = self.rotate_parts(parts_in_perpcone, -rotation_sign*np.pi/2)
    # for part in parts_in_perpcone:
    #   print('after rotation (pt, eta, phi)',part.pt(),part.eta(),part.phi())

    parts_in_cone = fj.vectorPJ()
    for part in parts_in_jet:
      part.set_user_index(999)
      parts_in_cone.append(part)
    for part in parts_in_perpcone:
      part.set_user_index(-999)
      parts_in_cone.append(part)

    return parts_in_cone

  def construct_parts_in_2perpcone(self, parts, jet, jetR, perpcone_R, rotation_sign):

    perpcone_R_effective = perpcone_R
    if self.do_rho_subtraction and self.static_perpcone == False and perpcone_R == jetR:
      perpcone_R_effective = math.sqrt(jet.area()/np.pi) # NB: for dynamic cone size

    # find perpcone at +pi/2 away and rotate it to jet direction
    perp_jet1 = fj.PseudoJet()
    perp_jet1.reset_PtYPhiM(jet.pt(), jet.rapidity(), jet.phi() + rotation_sign*np.pi/2, jet.m())
    parts_in_perpcone1 = self.find_parts_around_jet(parts, perp_jet1, perpcone_R_effective)
    parts_in_perpcone1 = self.rotate_parts(parts_in_perpcone1, -rotation_sign*np.pi/2)

    # perpcone at -pi/2 away and rotate it to jet direction
    perp_jet2 = fj.PseudoJet()
    perp_jet2.reset_PtYPhiM(jet.pt(), jet.rapidity(), jet.phi() - rotation_sign*np.pi/2, jet.m())
    parts_in_perpcone2 = self.find_parts_around_jet(parts, perp_jet2, perpcone_R_effective)
    parts_in_perpcone2 = self.rotate_parts(parts_in_perpcone2, +rotation_sign*np.pi/2)

    # label one perpcone as "sig" and the other as "bkg" so the perp1-perp2 and perp1(2)-perp1(2) correlations can be saved separately
    parts_in_cone = fj.vectorPJ()
    for part in parts_in_perpcone1:
      part.set_user_index(999)
      parts_in_cone.append(part)
    for part in parts_in_perpcone2:
      part.set_user_index(-999)
      parts_in_cone.append(part)

    return parts_in_cone

  def find_parts_around_jet(self, parts, jet, cone_R):
    # select particles around jet axis
    cone_parts = fj.vectorPJ()
    for part in parts:
      if jet.delta_R(part) <= cone_R:
        cone_parts.push_back(part)
    
    return cone_parts

  def rotate_parts(self, parts, rotate_phi):
    # rotate parts in azimuthal direction (NB: manually update the user index also)
    parts_rotated = fj.vectorPJ()
    for part in parts:
      pt_new = part.pt()
      y_new = part.rapidity()
      phi_new = part.phi() + rotate_phi
      m_new = part.m()
      user_index_new = part.user_index()
      # print('before',part.phi())
      part.reset_PtYPhiM(pt_new, y_new, phi_new, m_new)
      part.set_user_index(user_index_new)
      # print('after',part.phi())
      parts_rotated.push_back(part)
    
    return parts_rotated

  def copy_parts(self, parts, remove_ghosts = True):
    # don't need to re-init every part for a deep copy
    # the last arguement enable/disable the removal of ghost particles from jet area calculation (default set to true)
    parts_copied = fj.vectorPJ()
    for part in parts:
      # user_index_new = part.user_index()
      # part_new = fj.PseudoJet(part.px(), part.py(), part.pz(), part.E())
      # part_new.set_user_index(user_index_new)
      if remove_ghosts:
        if part.pt() > 0.01:
          parts_copied.push_back(part)
      else:
        parts_copied.push_back(part)
    
    return parts_copied

  #---------------------------------------------------------------
  # This function is called once
  # You must implement this
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
  
    raise NotImplementedError('You must implement initialize_user_output_objects()!')

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # You must implement this
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix):
  
    raise NotImplementedError('You must implement fill_jet_histograms()!')

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # You must implement this
  #---------------------------------------------------------------
  def fill_jet_cone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):
  
    raise NotImplementedError('You must implement fill_perp_cone_histograms()!')

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # You must implement this
  #---------------------------------------------------------------
  def fill_perp_cone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):
  
    raise NotImplementedError('You must implement fill_perp_cone_histograms()!')
