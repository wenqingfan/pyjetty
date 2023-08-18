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

    if 'do_jetcone' in config:
      self.do_jetcone = config['do_jetcone']
    else:
      self.do_jetcone = False
    if self.do_jetcone and 'jetcone_R' in config:
      self.jetcone_R = config['jetcone_R']
    else:
      self.jetcone_R = 0.4 # NB: set default value to 0.4

    if 'leading_pt' in config:
        self.leading_pt = config['leading_pt']
    else:
        self.leading_pt = -1 # negative means no leading track cut
    
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

    for R_max in self.max_distance:
      print('R_max',R_max,'alpha',self.alpha,'grid size',self.bge_rho_grid_size)
      
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
            self.analyze_jets(jets_selected_unsub, jetR, R_max = R_max, rho_bge = rho)
            if self.do_perpcone:
              self.analyze_perp_cones(fj_particles, jets_selected_unsub, jetR, R_max = R_max, rho_bge = rho)
            if self.do_jetcone:
              self.analyze_jet_cones(fj_particles, jets_selected_unsub, jetR, R_max = R_max, rho_bge = rho)
          else:
            self.analyze_jets(jets_selected, jetR, R_max = R_max)
            if self.do_perpcone:
              self.analyze_perp_cones(fj_particles, jets_selected, jetR, R_max = R_max)
            if self.do_jetcone:
              self.analyze_jet_cones(fj_particles, jets_selected, jetR, R_max = R_max)

  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets_selected, jetR, R_max = None, rho_bge = 0):
  
    # Set suffix for filling histograms
    if R_max:
      suffix = '_Rmax{}'.format(R_max)
    else:
      suffix = ''

    jets_reselected = fj.vectorPJ()
    for jet in jets_selected:
      is_jet_selected = True
      
      # leading track selection
      if self.leading_pt > 0:
        constituent = fj.sorted_by_pt(jet.constituents())
        if constituent[0] < self.leading_pt:
          is_jet_selected = False
      
      # if rho subtraction, require jet pt > 5 after subtration
      if self.do_rho_subtraction and rho_bge > 0:
        if jet.perp()-rho_bge*jet.area() < 5:
          # FIX ME: not sure whether to apply the area selection or not yet. jet.area() > 0.6*np.pi*jetR*jetR
          is_jet_selected = False

      if is_jet_selected:
        jets_reselected.append(jet)
    
    # Loop through jets and call user function to fill histos
    result = [self.analyze_accepted_jet(jet, jetR, suffix, rho_bge) for jet in jets_reselected]

  #---------------------------------------------------------------
  # Fill histograms
  #---------------------------------------------------------------
  def analyze_accepted_jet(self, jet, jetR, suffix, rho_bge = 0):
    
    # Check additional acceptance criteria
    if not self.utils.is_det_jet_accepted(jet):
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

  def find_particles_in_cone(self, parts, cone_center_phi, cone_center_eta, cone_R):
    # select particles around cone center
    # conver cone center phi to [0, 2pi]
    if cone_center_phi > 2*np.pi:
        cone_center_phi = cone_center_phi - 2*np.pi
    if cone_center_phi < 0:
        cone_center_phi = cone_center_phi + 2*np.pi
    
    # print('cone R',cone_R,'phi',cone_center_phi,'eta',cone_center_eta,'area',np.pi*cone_R*cone_R)
    cone_parts = fj.vectorPJ()
    for part in parts:
      dphi = part.phi()-cone_center_phi
      if dphi > 2*np.pi:
        dphi = dphi - 2*np.pi 
      if dphi < 0:
        dphi = dphi + 2*np.pi 
      deta = part.eta()-cone_center_eta
      if math.sqrt(dphi*dphi+deta*deta) <= cone_R:
        cone_parts.push_back(part)
    
    return cone_parts

  def find_parts_around_jets(self, parts, jet, cone_R):
    # select particles around jet axis
    parts = fj.vectorPJ()
    for part in parts:
      if jet.delta_R(part) <= cone_R:
        parts.push_back(part)
    
    return parts

  def rotate_parts(self, parts, rotate_phi):
    # rotate parts in azimuthal direction
    parts_rotated = fj.vectorPJ()
    for part in parts:
      pt_new = part.pt()
      y_new = part.rapidity()
      phi_new = part.phi() + rotate_phi
      m_new = part.m()
      # print('before',part.phi())
      part.reset_PtYPhiM(pt_new, y_new, phi_new, m_new)
      # print('after',part.phi())
      parts_rotated.push_back(part)
    
    return parts_rotated

  def analyze_jet_cones(self, parts, jets_selected, jetR, R_max = None, rho_bge = 0):
    # analyze cones around jet axis. NB: maybe can even use WTA axis
    if R_max:
      suffix = '_Rmax{}'.format(R_max)
    else:
      suffix = ''

    # NB: set cone R to be bigger than jet R
    cone_R = self.jetcone_R
    if self.do_rho_subtraction and rho_bge > 0:
      jets_reselected = []
      for jet in jets_selected:
        if jet.perp()-rho_bge*jet.area() > 5:
          jets_reselected.append(jet)

      for jet in jets_reselected:
        parts_in_cone = self.find_parts_around_jets(parts, jet, cone_R)
        self.analyze_accepted_cone(False, parts_in_cone, jet, jetR, suffix, rho_bge)
    else:
      for jet in jets_selected:
        parts_in_cone = self.find_parts_around_jets(parts, jet, cone_R)
        self.analyze_accepted_cone(False, parts_in_cone, jet, jetR, suffix, rho_bge)

  def analyze_perp_cones(self, parts, jets_selected, jetR, R_max = None, rho_bge = 0):
    # analyze cones perpendicular to jet in the azimuthal plane
    if R_max:
      suffix = '_Rmax{}'.format(R_max)
    else:
      suffix = ''

    # NB: initialize cone R using jet R. Later update using area if area is available
    cone_R = jetR
    if self.do_rho_subtraction and rho_bge > 0:
      jets_reselected = []
      for jet in jets_selected:
        if jet.perp()-rho_bge*jet.area() > 5:
          jets_reselected.append(jet)

      for jet in jets_reselected:
        # print('jet pt',jet.perp()-rho_bge*jet.area(),'phi',jet.phi(),'eta',jet.eta(),'area',jet.area())
        cone_perp_phi1 = jet.phi() + np.pi/2
        cone_perp_phi2 = jet.phi() - np.pi/2
        cone_perp_eta = jet.eta()
        cone_R = math.sqrt(jet.area()/np.pi) # NB: jet area is available only when rho subtraction flag is on
        parts_in_cone1 = self.find_particles_in_cone(parts, cone_perp_phi1, cone_perp_eta, cone_R)
        # for part in parts_in_cone1:
        #   print('before rotation',part.phi())
        parts_in_cone1 = self.rotate_parts(parts_in_cone1, -np.pi/2)
        # for part in parts_in_cone1:
        #   print('after rotation',part.phi())
        parts_in_cone2 = self.find_particles_in_cone(parts, cone_perp_phi2, cone_perp_eta, cone_R)
        parts_in_cone2 = self.rotate_parts(parts_in_cone2, +np.pi/2)
        
        self.analyze_accepted_cone(True, parts_in_cone1, jet, jetR, suffix, rho_bge)
        self.analyze_accepted_cone(True, parts_in_cone2, jet, jetR, suffix, rho_bge)
    else:
      for jet in jets_selected:
        cone_perp_phi1 = jet.phi() + np.pi/2
        cone_perp_phi2 = jet.phi() - np.pi/2
        cone_perp_eta = jet.eta()
        parts_in_cone1 = self.find_particles_in_cone(parts, cone_perp_phi1, cone_perp_eta, cone_R)
        parts_in_cone1 = self.rotate_parts(parts_in_cone1, -np.pi/2)
        parts_in_cone2 = self.find_particles_in_cone(parts, cone_perp_phi2, cone_perp_eta, cone_R)
        parts_in_cone2 = self.rotate_parts(parts_in_cone2, +np.pi/2)
        
        self.analyze_accepted_cone(True, parts_in_cone1, jet, jetR, suffix, rho_bge)
        self.analyze_accepted_cone(True, parts_in_cone2, jet, jetR, suffix, rho_bge)

  def analyze_accepted_cone(self, is_perp, cone_parts, jet, jetR, suffix, rho_bge = 0):
    
    # Check additional acceptance criteria
    if not self.utils.is_det_jet_accepted(jet):
      return
          
    # Fill base histograms
    if self.do_rho_subtraction and rho_bge > 0:
      jet_pt_ungroomed = jet.pt() - rho_bge*jet.area()
    else:
      jet_pt_ungroomed = jet.pt()
    
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
      if is_perp:
        self.fill_perp_cone_histograms(cone_parts, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                               obs_label, jet_pt_ungroomed, suffix, rho_bge)
      else:
        self.fill_jet_cone_histograms(cone_parts, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                               obs_label, jet_pt_ungroomed, suffix, rho_bge)

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
  def fill_perp_cone_histograms(self, cone_parts, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):
  
    raise NotImplementedError('You must implement fill_perp_cone_histograms()!')
