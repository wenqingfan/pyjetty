#!/usr/bin/env python3

"""
Base class to read a ROOT TTree of track information
and do jet-finding, and save basic histograms.
  
To use this class, the following should be done:

  - Implement a user analysis class inheriting from this one, such as in user/james/process_mc_XX.py
    You should implement the following functions:
      - initialize_user_output_objects_R()
      - fill_observable_histograms()
      - fill_matched_jet_histograms()
    
  - You should include the following histograms:
      - Response matrix: hResponse_JetPt_[obs]_R[R]_[subobs]_[grooming setting]
      - Residual distribution: hResidual_JetPt_[obs]_R[R]_[subobs]_[grooming setting]

  - You also should modify observable-specific functions at the top of common_utils.py
  
Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import time

# Data analysis and plotting
import pandas
import numpy as np
from array import *
import ROOT
import yaml
import random
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjtools

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_io_emb
from pyjetty.alice_analysis.process.base import process_base
from pyjetty.alice_analysis.process.base import thermal_generator
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.mputils.csubtractor import CEventSubtractor

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class ProcessMCBase(process_base.ProcessBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMCBase, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
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
      
    self.fast_simulation = config['fast_simulation']
    if self.fast_simulation == True:
      if 'ENC_fastsim' in config:
          self.ENC_fastsim = config['ENC_fastsim']
      else:
          self.ENC_fastsim = False
    else: # if not fast simulation, set ENC_fastsim flag to False
      self.ENC_fastsim = False  
    if self.ENC_fastsim == True:
      self.pair_eff_file = config['pair_eff_file'] # load pair efficiency input for fastsim
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
    if 'jetscape' in config:
        self.jetscape = config['jetscape']
    else:
        self.jetscape = False
    if 'event_plane_angle' in config:
      self.event_plane_range = config['event_plane_angle']
    else:
      self.event_plane_range = None
    if 'matching_systematic' in config:
      self.matching_systematic = config['matching_systematic']
    else:
      self.matching_systematic = False
    self.dry_run = config['dry_run']
    self.skip_deltapt_RC_histograms = True
    self.fill_RM_histograms = True
    
    self.jet_matching_distance = config['jet_matching_distance']
    self.reject_tracks_fraction = config['reject_tracks_fraction']
    if 'mc_fraction_threshold' in config:
      self.mc_fraction_threshold = config['mc_fraction_threshold']
    if 'do_rho_subtraction' in config:
      self.do_rho_subtraction = config['do_rho_subtraction']
    else:
      self.do_rho_subtraction = False
    if 'do_jetcone' in config:
      self.do_jetcone = config['do_jetcone']
    else:
      self.do_jetcone = False
    if self.do_jetcone and 'jetcone_R_list' in config:
      self.jetcone_R_list = config['jetcone_R_list']
    else:
      self.jetcone_R_list = [0.4] # NB: set default value to 0.4
    
    # right now donot allow perpcone and jetcone to be enabled as the same time
    if ('do_perpcone' in config) and (self.do_jetcone==False):
      self.do_perpcone = config['do_perpcone']
    else:
      self.do_perpcone = False

    if 'leading_pt' in config:
        self.leading_pt = config['leading_pt']
    else:
        self.leading_pt = -1 # negative means no leading track cut
    
    if self.do_constituent_subtraction:
        self.is_pp = False
        self.emb_file_list = config['emb_file_list']
        self.main_R_max = config['constituent_subtractor']['main_R_max']
    else:
        self.is_pp = True
        
    if 'thermal_model' in config:
      self.thermal_model = True
      beta = config['thermal_model']['beta']
      N_avg = config['thermal_model']['N_avg']
      sigma_N = config['thermal_model']['sigma_N']
      self.thermal_generator = thermal_generator.ThermalGenerator(N_avg, sigma_N, beta)
    else:
      self.thermal_model = False

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
  def process_mc(self):
    
    self.start_time = time.time()
    
    # ------------------------------------------------------------------------
    
    # Use IO helper class to convert detector-level ROOT TTree into
    # a SeriesGroupBy object of fastjet particles per event
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    if self.fast_simulation:
      tree_dir = ''
    else:
      tree_dir = 'PWGHF_TreeCreator'
    io_det = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                  track_tree_name='tree_Particle', use_ev_id_ext=False,
                                  is_jetscape=self.jetscape, event_plane_range=self.event_plane_range, is_ENC=self.ENC_fastsim, is_det_level=True)
    df_fjparticles_det = io_det.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
    self.nEvents_det = len(df_fjparticles_det.index)
    self.nTracks_det = len(io_det.track_df.index)
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # If jetscape, store also the negative status particles (holes)
    if self.jetscape:
        io_det_holes = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                           track_tree_name='tree_Particle', use_ev_id_ext=False,
                                            is_jetscape=self.jetscape, holes=True,
                                            event_plane_range=self.event_plane_range)
        df_fjparticles_det_holes = io_det_holes.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
        self.nEvents_det_holes = len(df_fjparticles_det_holes.index)
        self.nTracks_det_holes = len(io_det_holes.track_df.index)
        print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # ------------------------------------------------------------------------

    # Use IO helper class to convert truth-level ROOT TTree into
    # a SeriesGroupBy object of fastjet particles per event
    io_truth = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                    track_tree_name='tree_Particle_gen', use_ev_id_ext=False,
                                    is_jetscape=self.jetscape, event_plane_range=self.event_plane_range, is_ENC=self.ENC_fastsim, is_det_level=False)
    df_fjparticles_truth = io_truth.load_data(m=self.m) # no dropping of tracks at truth level (important for the det-truth association because the index of the truth particle is used)
    self.nEvents_truth = len(df_fjparticles_truth.index)
    self.nTracks_truth = len(io_truth.track_df.index)
    print('--- {} seconds ---'.format(time.time() - self.start_time))

    print('Input truth Data Frame',df_fjparticles_truth)
    
    # If jetscape, store also the negative status particles (holes)
    if self.jetscape:
        io_truth_holes = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                              track_tree_name='tree_Particle_gen', use_ev_id_ext=False,
                                              is_jetscape=self.jetscape, holes=True,
                                              event_plane_range=self.event_plane_range)
        df_fjparticles_truth_holes = io_truth_holes.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
        self.nEvents_truth_holes = len(df_fjparticles_truth_holes.index)
        self.nTracks_truth_holes = len(io_truth_holes.track_df.index)
        print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # ------------------------------------------------------------------------

    # Now merge the two SeriesGroupBy to create a groupby df with [ev_id, run_number, fj_1, fj_2]
    # (Need a structure such that we can iterate event-by-event through both fj_1, fj_2 simultaneously)
    # In the case of jetscape, we merge also the hole collections fj_3, fj_4
    print('Merge det-level and truth-level into a single dataframe grouped by event...')
    # print('debug df_fjparticles_det',df_fjparticles_det)
    # print('debug df_fjparticles_truth',df_fjparticles_truth)
    if self.jetscape:
        self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth, df_fjparticles_det_holes, df_fjparticles_truth_holes], axis=1)
        self.df_fjparticles.columns = ['fj_particles_det', 'fj_particles_truth', 'fj_particles_det_holes', 'fj_particles_truth_holes']
    elif self.ENC_fastsim:
        self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth], axis=1)
        self.df_fjparticles.columns = ['fj_particles_det', 'ParticleMCIndex', 'fj_particles_truth', 'ParticlePID']
        print('Merged output',self.df_fjparticles.columns)
        print(self.df_fjparticles)
    else:
        self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth], axis=1)
        self.df_fjparticles.columns = ['fj_particles_det', 'fj_particles_truth']
    print('--- {} seconds ---'.format(time.time() - self.start_time))

    # ------------------------------------------------------------------------
    
    # Set up the Pb-Pb embedding object
    if not self.is_pp and not self.thermal_model:
        self.process_io_emb = process_io_emb.ProcessIO_Emb(self.emb_file_list, track_tree_name='tree_Particle', m=self.m)
    
    # ------------------------------------------------------------------------

    # Initialize histograms
    if not self.dry_run:
      self.initialize_output_objects()
    
    # Create constituent subtractor, if configured
    if self.do_constituent_subtraction:
      self.constituent_subtractor = [CEventSubtractor(max_distance=R_max, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR) for R_max in self.max_distance]
    
    print(self)
    
    # Find jets and fill histograms
    print('Find jets...')
    self.analyze_events()
    
    # Plot histograms
    print('Save histograms...')
    process_base.ProcessBase.save_output_objects(self)
    
    print('--- {} seconds ---'.format(time.time() - self.start_time))

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_output_objects(self):
    
    self.hNevents = ROOT.TH1F('hNevents', 'hNevents', 2, -0.5, 1.5)
    self.hNevents.Fill(1, self.nEvents_det)
    
    self.hTrackEtaPhi = ROOT.TH2F('hTrackEtaPhi', 'hTrackEtaPhi', 200, -1., 1., 628, 0., 6.28)
    self.hTrackPt = ROOT.TH1F('hTrackPt', 'hTrackPt', 300, 0., 300.)
    
    if not self.is_pp:
      self.hRho =  ROOT.TH1F('hRho', 'hRho', 1000, 0., 1000.)
      
    if not self.skip_deltapt_RC_histograms:
      name = 'hN_MeanPt'
      h = ROOT.TH2F(name, name, 200, 0, 5000, 200, 0., 2.)
      setattr(self, name, h)

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_output_objects_R(self, jetR):
  
      # Call user-specific initialization
      self.initialize_user_output_objects_R(jetR)
      
      # Base histograms
      if self.is_pp:
      
          name = 'hJES_R{}'.format(jetR)
          h = ROOT.TH2F(name, name, 300, 0, 300, 200, -1., 1.)
          setattr(self, name, h)
      
          name = 'hDeltaR_All_R{}'.format(jetR)
          h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
          setattr(self, name, h)
          
      else:
      
          for R_max in self.max_distance:
          
            name = 'hJES_R{}_Rmax{}'.format(jetR, R_max)
            h = ROOT.TH2F(name, name, 300, 0, 300, 200, -1., 1.)
            setattr(self, name, h)
          
            name = 'hDeltaPt_emb_R{}_Rmax{}'.format(jetR, R_max)
            h = ROOT.TH2F(name, name, 300, 0, 300, 400, -200., 200.)
            setattr(self, name, h)
            
            if not self.skip_deltapt_RC_histograms:
              name = 'hDeltaPt_RC_beforeCS_R{}_Rmax{}'.format(jetR, R_max)
              h = ROOT.TH1F(name, name, 400, -200., 200.)
              setattr(self, name, h)
              
              name = 'hDeltaPt_RC_afterCS_R{}_Rmax{}'.format(jetR, R_max)
              h = ROOT.TH1F(name, name, 400, -200., 200.)
              setattr(self, name, h)
      
            name = 'hDeltaR_ppdet_pptrue_R{}_Rmax{}'.format(jetR, R_max)
            h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
            setattr(self, name, h)
            
            name = 'hDeltaR_combined_ppdet_R{}_Rmax{}'.format(jetR, R_max)
            h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
            setattr(self, name, h)
              
      name = 'hZ_Truth_R{}'.format(jetR)
      h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 1.)
      setattr(self, name, h)
      
      name = 'hZ_Det_R{}'.format(jetR)
      h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 1.)
      setattr(self, name, h)

  #---------------------------------------------------------------
  # Main function to loop through and analyze events
  #---------------------------------------------------------------
  def analyze_events(self):
    
    # Fill track histograms
    if not self.dry_run:
      [self.fill_track_histograms(fj_particles_det) for fj_particles_det in self.df_fjparticles['fj_particles_det']]
    
    fj.ClusterSequence.print_banner()
    print()
        
    self.event_number = 0
    
    for jetR in self.jetR_list:
      if not self.dry_run:
        self.initialize_output_objects_R(jetR)
    
    # Then can use list comprehension to iterate over the groupby and do jet-finding
    # simultaneously for fj_1 and fj_2 per event, so that I can match jets -- and fill histograms
    if self.jetscape:
        result = [self.analyze_event(fj_particles_det, fj_particles_truth, fj_particles_det_holes, fj_particles_truth_holes) for fj_particles_det, fj_particles_truth, fj_particles_det_holes, fj_particles_truth_holes in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'], self.df_fjparticles['fj_particles_det_holes'], self.df_fjparticles['fj_particles_truth_holes'])]
    elif self.ENC_fastsim:
        result = [self.analyze_event(fj_particles_det=fj_particles_det, fj_particles_truth=fj_particles_truth, particles_mcid_det=particles_mcid_det, particles_pid_truth=particles_pid_truth) for fj_particles_det, fj_particles_truth, particles_mcid_det, particles_pid_truth in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'], self.df_fjparticles['ParticleMCIndex'], self.df_fjparticles['ParticlePID'])]
    else:
        result = [self.analyze_event(fj_particles_det, fj_particles_truth) for fj_particles_det, fj_particles_truth in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'])]
    
    if self.debug_level > 0:
      for attr in dir(self):
        obj = getattr(self, attr)
        print('size of {}: {}'.format(attr, sys.getsizeof(obj)))
        
    print('Save thn...')
    process_base.ProcessBase.save_thn_th3_objects(self)
    
  #---------------------------------------------------------------
  # Fill track histograms.
  #---------------------------------------------------------------
  def fill_track_histograms(self, fj_particles_det):

    # Check that the entries exist appropriately
    # (need to check how this can happen -- but it is only a tiny fraction of events)
    if type(fj_particles_det) != fj.vectorPJ:
      return
    
    for track in fj_particles_det:
      self.hTrackEtaPhi.Fill(track.eta(), track.phi())
      self.hTrackPt.Fill(track.pt())
      
  #---------------------------------------------------------------
  # Analyze jets of a given event.
  # fj_particles is the list of fastjet pseudojets for a single fixed event.
  #---------------------------------------------------------------
  def analyze_event(self, fj_particles_det, fj_particles_truth, fj_particles_det_holes=None, fj_particles_truth_holes=None, particles_mcid_det=None, particles_pid_truth=None):
  
    self.event_number += 1
    if self.event_number > self.event_number_max:
      return
    if self.debug_level > 1:
      print('-------------------------------------------------')
      print('event {}'.format(self.event_number))

    # print('debug5 det parts',fj_particles_det)
    # print('debug5 mcid',particles_mcid_det)
    # print('debug5 truth parts',fj_particles_truth)
    # print('debug5 pid',particles_pid_truth)

    if self.ENC_fastsim:
      # make charge array from pid info, needed for pair efficiency determination
        particles_charge_truth = np.array([])
        for pid in particles_pid_truth:
            # charged hadrons
            if abs(pid)==211 or abs(pid)==321 or abs(pid)==2212 or abs(pid)==3222:
                if pid>0:
                    particles_charge_truth = np.append(particles_charge_truth, 1)
                else:
                    particles_charge_truth = np.append(particles_charge_truth, -1)
            # electrons and muons
            elif abs(pid)==11 or abs(pid)==13 or abs(pid)==3112 or abs(pid)==3312 or abs(pid)==3334:
                if pid>0:
                    particles_charge_truth = np.append(particles_charge_truth, -1)
                else:
                    particles_charge_truth = np.append(particles_charge_truth, 1)
            # long lived weak decay particles (<2% of the total number of charged particles)
            # for now mark as charge 0 and later NOT applying pair efficiency for 0-charged or 0-0 pairs
            # NB: this can be avoided by decaying these paritcles within the generation step
            else:
                particles_charge_truth = np.append(particles_charge_truth, 0)
      
    # Check that the entries exist appropriately
    # (need to check how this can happen -- but it is only a tiny fraction of events)
    if type(fj_particles_det) != fj.vectorPJ or type(fj_particles_truth) != fj.vectorPJ:
      print('fj_particles type mismatch -- skipping event')
      return
    else:
      # Todo
      ## for full simulation, match det-level and truth level particles
      ## sort both list by pT, phi and eta first before matching

      # add associated truth info and charge info in fj_particles_det using the JetInfo object
      if self.ENC_fastsim:
        for index, mcid in enumerate(particles_mcid_det):
          if fj_particles_det[index].has_user_info():
            ecorr_user_info = fj_particles_det[index].python_info()
          else:
            ecorr_user_info = jet_info.JetInfo()
          if mcid>=0 and mcid<len(fj_particles_truth):
            # print('debug6', p, 'mcid/length', mcid, len(fj_particles_truth))
            # print('debug6', p, 'truth', fj_particles_truth[int(mcid)])
            # print('debug6', p, 'charge', particles_charge_truth[int(mcid)])
            ecorr_user_info.particle_truth = fj_particles_truth[int(mcid)]
            ecorr_user_info.charge = particles_charge_truth[int(mcid)]
          else:
            print("invalid associated MC Index, filling default values (particle_truth = None, charge = 1000)")
          fj_particles_det[index].set_python_info(ecorr_user_info)
          # fj_particles_det[index].set_user_index(int(mcid))

        for index in range( len(fj_particles_truth) ):
          if fj_particles_truth[index].has_user_info():
            ecorr_user_info = fj_particles_truth[index].python_info()
          else:
            ecorr_user_info = jet_info.JetInfo()
          ecorr_user_info.particle_truth = fj_particles_truth[index]
          ecorr_user_info.charge = particles_charge_truth[index]
          fj_particles_truth[index].set_python_info(ecorr_user_info)
          # fj_particles_truth[index].set_user_index(int(index))

    if self.jetscape:
      if type(fj_particles_det_holes) != fj.vectorPJ or type(fj_particles_truth_holes) != fj.vectorPJ:
        print('fj_particles_holes type mismatch -- skipping event')
        return
      
    if len(fj_particles_truth) > 1:
      if np.abs(fj_particles_truth[0].pt() - fj_particles_truth[1].pt()) <  1e-10:
        print('WARNING: Duplicate particles may be present')
        print([p.user_index() for p in fj_particles_truth])
        print([p.pt() for p in fj_particles_truth])

    # If Pb-Pb, construct embedded event (do this once, for all jetR)
    if not self.is_pp:
        
        # If thermal model, generate a thermal event and add it to the det-level particle list
        if self.thermal_model:
          fj_particles_combined_beforeCS = self.thermal_generator.load_event()
          
          # Form the combined det-level event
          # The pp-det tracks are each stored with a unique user_index >= 0
          #   (same index in fj_particles_combined and fj_particles_det -- which will be used in prong-matching)
          # The thermal tracks are each stored with a unique user_index < 0
          [fj_particles_combined_beforeCS.push_back(p) for p in fj_particles_det]

        # Main case: Get Pb-Pb event and embed it into the det-level particle list
        else:
          fj_particles_combined_beforeCS = self.process_io_emb.load_event()
              
          # Form the combined det-level event
          # The pp-det tracks are each stored with a unique user_index >= 0
          #   (same index in fj_particles_combined and fj_particles_det -- which will be used in prong-matching)
          # The Pb-Pb tracks are each stored with a unique user_index < 0
          [fj_particles_combined_beforeCS.push_back(p) for p in fj_particles_det]
         
        # Perform constituent subtraction for each R_max
        fj_particles_combined = [self.constituent_subtractor[i].process_event(fj_particles_combined_beforeCS) for i, R_max in enumerate(self.max_distance)]
        # for i, R_max in enumerate(self.max_distance):
        #   rho = self.constituent_subtractor[i].bge_rho.rho()
        #   print('rho is ',rho)
        # print('**************Before CS subtraction*****************')
        # n_sig_before = 0
        # n_bkg_before = 0
        # for part in fj_particles_combined_beforeCS:
        #   if part.user_index() < 0:
        #     n_bkg_before += 1
        #   else:
        #     n_sig_before += 1
        #     print('index checking:',part.user_index(),'pt',part.perp(),'phi',part.phi(),'eta',part.eta())
        # print('n_sig',n_sig_before,'n_bkg',n_bkg_before)
        # print('**************After CS subtraction*****************')
        # n_sig_after = 0
        # n_bkg_after = 0
        # for part in fj_particles_combined[0]:
        #   if part.user_index() < 0:
        #     n_bkg_after += 1
        #   else:
        #     n_sig_after += 1
        #     print('index checking:',part.user_index(),'pt',part.perp(),'phi',part.phi(),'eta',part.eta())
        # print('n_sig',n_sig_after,'n_bkg',n_bkg_after)
        # print('**************After CS subtraction*****************')
        
        if self.debug_level > 3:
          print([p.user_index() for p in fj_particles_truth])
          print([p.pt() for p in fj_particles_truth])
          print([p.user_index() for p in fj_particles_det])
          print([p.pt() for p in fj_particles_det])
          print([p.user_index() for p in fj_particles_combined_beforeCS])
          print([p.pt() for p in fj_particles_combined_beforeCS])
          
    if self.dry_run:
      return

    # Loop through jetR, and process event for each R
    for jetR in self.jetR_list:        
    
      # Keep track of whether to fill R-independent histograms
      self.fill_R_indep_hists = (jetR == self.jetR_list[0])

      # Set jet definition and a jet selector
      jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
      jet_selector_det = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR)
      jet_selector_truth_matched = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9)
      if self.debug_level > 2:
        print('')
        print('jet definition is:', jet_def)
        print('jet selector for det-level is:', jet_selector_det)
        print('jet selector for truth-level matches is:', jet_selector_truth_matched)
      
      # Analyze
      if self.is_pp:

        # Find pp det and truth jets
        if self.ENC_fastsim:
          # FIX ME: should treat long lived charged particle differently (check how the existing fast herwig and pythia handles it)
          fj_particles_det_ch = fj.vectorPJ()
          for part in fj_particles_det:
            if part.python_info().charge!=0: # only use charged particles
              fj_particles_det_ch.append(part)
          cs_det = fj.ClusterSequence(fj_particles_det_ch, jet_def)
        else:
          cs_det = fj.ClusterSequence(fj_particles_det, jet_def)
        
        jets_det_pp = fj.sorted_by_pt(cs_det.inclusive_jets())
        # make sure the user info (on the jet side) for jets are all empty right after the jet-clustering 
        for jet in jets_det_pp:
          if jet.has_user_info():
            jet.python_info().clear_jet_info()
        jets_det_pp_selected = jet_selector_det(jets_det_pp)
        
        if self.ENC_fastsim:
          # FIX ME: should treat long lived charged particle differently (check how the existing fast herwig and pythia handles it)
          fj_particles_truth_ch = fj.vectorPJ()
          for part in fj_particles_truth:
            if part.python_info().charge!=0: # only use charged particles
              fj_particles_truth_ch.append(part)
          cs_truth = fj.ClusterSequence(fj_particles_truth_ch, jet_def)
        else:
          cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)

        jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
        # make sure the user info (on the jet side) for jets are all empty right after the jet-clustering  
        for jet in jets_truth:
          if jet.has_user_info():
            jet.python_info().clear_jet_info()
        jets_truth_selected = jet_selector_det(jets_truth)
        jets_truth_selected_matched = jet_selector_truth_matched(jets_truth)
      
        self.analyze_jets(jets_det_pp_selected, jets_truth_selected, jets_truth_selected_matched, jetR)
        
      else:
      
        for i, R_max in enumerate(self.max_distance):
            
          if self.debug_level > 1:
            print('')
            print('R_max: {}'.format(R_max))
            print('Total number of combined particles: {}'.format(len([p.pt() for p in fj_particles_combined_beforeCS])))
            print('After constituent subtraction {}: {}'.format(i, len([p.pt() for p in fj_particles_combined[i]])))
            
          # Keep track of whether to fill R_max-independent histograms
          self.fill_Rmax_indep_hists = (i == 0)
          
          # Perform constituent subtraction on det-level, if applicable
          self.fill_background_histograms(fj_particles_combined_beforeCS, fj_particles_combined[i], jetR, i)
          rho = self.constituent_subtractor[i].bge_rho.rho() 
      
          # Do jet finding (re-do each time, to make sure matching info gets reset)
          cs_det = fj.ClusterSequence(fj_particles_det, jet_def)
          jets_det_pp = fj.sorted_by_pt(cs_det.inclusive_jets())
          jets_det_pp_selected = jet_selector_det(jets_det_pp)
          
          cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)
          jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
          jets_truth_selected = jet_selector_det(jets_truth)
          jets_truth_selected_matched = jet_selector_truth_matched(jets_truth)
          
          cs_combined = fj.ClusterSequence(fj_particles_combined[i], jet_def)
          jets_combined = fj.sorted_by_pt(cs_combined.inclusive_jets())
          jets_combined_selected = jet_selector_det(jets_combined)

          if self.do_rho_subtraction:
            cs_combined_beforeCS = fj.ClusterSequenceArea(fj_particles_combined_beforeCS, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
            jets_combined_beforeCS = fj.sorted_by_pt(cs_combined_beforeCS.inclusive_jets())
            jets_combined_selected_beforeCS = jet_selector_det(jets_combined_beforeCS)

            jets_combined_reselected_beforeCS = self.reselect_jets(jets_combined_selected_beforeCS, jetR, rho_bge = rho)

            if self.do_jetcone or self.do_perpcone:
            # NB: either jetcone or perpcone (not both)
              self.analyze_jets(jets_combined_reselected_beforeCS, jets_truth_selected, jets_truth_selected_matched, jetR,
                            jets_det_pp_selected = jets_det_pp_selected, R_max = R_max,
                            fj_particles_det_holes = fj_particles_det_holes,
                            fj_particles_truth_holes = fj_particles_truth_holes, rho_bge = rho, fj_particles_det_cones = fj_particles_combined_beforeCS, fj_particles_truth_cones = fj_particles_truth)
            else:
              self.analyze_jets(jets_combined_reselected_beforeCS, jets_truth_selected, jets_truth_selected_matched, jetR,
                            jets_det_pp_selected = jets_det_pp_selected, R_max = R_max,
                            fj_particles_det_holes = fj_particles_det_holes,
                            fj_particles_truth_holes = fj_particles_truth_holes, rho_bge = rho)
          else:
            if self.do_jetcone or self.do_perpcone:
            # NB: either jetcone or perpcone (not both)
              self.analyze_jets(jets_combined_selected, jets_truth_selected, jets_truth_selected_matched, jetR,
                            jets_det_pp_selected = jets_det_pp_selected, R_max = R_max,
                            fj_particles_det_holes = fj_particles_det_holes,
                            fj_particles_truth_holes = fj_particles_truth_holes, rho_bge = 0, fj_particles_det_cones = fj_particles_combined_beforeCS, fj_particles_truth_cones = fj_particles_truth) # NB: feed all particles for cone around the CS subtracted jet. An alternate way is to use CS subtracted particles
            else:
              self.analyze_jets(jets_combined_selected, jets_truth_selected, jets_truth_selected_matched, jetR,
                            jets_det_pp_selected = jets_det_pp_selected, R_max = R_max,
                            fj_particles_det_holes = fj_particles_det_holes,
                            fj_particles_truth_holes = fj_particles_truth_holes, rho_bge = 0)

  #---------------------------------------------------------------
  # Jet selection cuts.
  #---------------------------------------------------------------
  def reselect_jets(self, jets_selected, jetR, rho_bge = 0):
    # re-apply jet pt > 5GeV cut after rho subtraction and leading track pt cut if there is any. NB: need to be applied inside apply_events to make sure matching work properly
    jets_reselected = []
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
  def analyze_jets(self, jets_det_selected, jets_truth_selected, jets_truth_selected_matched, jetR,
                   jets_det_pp_selected = None, R_max = None,
                   fj_particles_det_holes = None, fj_particles_truth_holes = None, rho_bge = 0, fj_particles_det_cones = None, fj_particles_truth_cones = None):
  
    if self.debug_level > 1:
      print('Number of det-level jets: {}'.format(len(jets_det_selected)))
    
    # Fill det-level jet histograms (before matching)
    for jet_det in jets_det_selected:
      
      # Check additional acceptance criteria
      # skip event if not satisfied -- since first jet in event is highest pt
      if not self.utils.is_det_jet_accepted(jet_det):
        if self.fill_R_indep_hists:
          self.hNevents.Fill(0)
        if self.debug_level > 1:
          print('event rejected due to jet acceptance')
        return
      
      self.fill_det_before_matching(jet_det, jetR, R_max, rho_bge)
  
    # Fill truth-level jet histograms (before matching)
    for jet_truth in jets_truth_selected:
    
      if self.is_pp or self.fill_Rmax_indep_hists:
        self.fill_truth_before_matching(jet_truth, jetR)
  
    # Loop through jets and set jet matching candidates for each jet in user_info
    if self.is_pp:
        [[self.set_matching_candidates(jet_det, jet_truth, jetR, 'hDeltaR_All_R{}'.format(jetR)) for jet_truth in jets_truth_selected_matched] for jet_det in jets_det_selected]
    else:
        # First fill the combined-to-pp matches, then the pp-to-pp matches
        [[self.set_matching_candidates(jet_det_combined, jet_det_pp, jetR, 'hDeltaR_combined_ppdet_R{{}}_Rmax{}'.format(R_max), fill_jet1_matches_only=True) for jet_det_pp in jets_det_pp_selected] for jet_det_combined in jets_det_selected]
        [[self.set_matching_candidates(jet_det_pp, jet_truth, jetR, 'hDeltaR_ppdet_pptrue_R{{}}_Rmax{}'.format(R_max)) for jet_truth in jets_truth_selected_matched] for jet_det_pp in jets_det_pp_selected]

    # # debug
    # for jet_det_combined in jets_det_selected:
    #   print('debug7.1--jet_det',jet_det_combined.pt(),'user_info',jet_det_combined.has_user_info())
    #   if jet_det_combined.has_user_info() and jet_det_combined.python_info().closest_jet:
    #     print('matches to',jet_det_combined.python_info().closest_jet.pt())
    #     print('debug7.1--jet_det',len(jet_det_combined.constituents()))
    #     print('matches to',len(jet_det_combined.python_info().closest_jet.constituents()))
        
    # Loop through jets and set accepted matches
    if self.is_pp:
        hname = 'hJetMatchingQA_R{}'.format(jetR)
        [self.set_matches_pp(jet_det, hname) for jet_det in jets_det_selected]
    else:
        hname = 'hJetMatchingQA_R{}_Rmax{}'.format(jetR, R_max)
        [self.set_matches_AA(jet_det_combined, jetR, hname) for jet_det_combined in jets_det_selected]
          
    # Loop through jets and fill response histograms if both det and truth jets are unique match
    result = [self.fill_jet_matches(jet_det, jetR, R_max, fj_particles_det_holes, fj_particles_truth_holes, rho_bge, fj_particles_det_cones, fj_particles_truth_cones) for jet_det in jets_det_selected]

  #---------------------------------------------------------------
  # Fill some background histograms
  #---------------------------------------------------------------
  def fill_background_histograms(self, fj_particles_combined_beforeCS, fj_particles_combined, jetR, i):

    # Fill rho
    rho = self.constituent_subtractor[i].bge_rho.rho()
    if self.fill_R_indep_hists and self.fill_Rmax_indep_hists:
      getattr(self, 'hRho').Fill(rho)
    
    # Fill random cone delta-pt before constituent subtraction
    if not self.skip_deltapt_RC_histograms:
      R_max = self.max_distance[i]
      self.fill_deltapt_RC_histogram(fj_particles_combined_beforeCS, rho, jetR, R_max, before_CS=True)
          
      # Fill random cone delta-pt after constituent subtraction
      self.fill_deltapt_RC_histogram(fj_particles_combined, rho, jetR, R_max, before_CS=False)
    
  #---------------------------------------------------------------
  # Fill delta-pt histogram
  #---------------------------------------------------------------
  def fill_deltapt_RC_histogram(self, fj_particles, rho, jetR, R_max, before_CS=False):
  
    # Choose a random eta-phi in the fiducial acceptance
    phi = random.uniform(0., 2*np.pi)
    eta = random.uniform(-0.9+jetR, 0.9-jetR)
    
    # Loop through tracks and sum pt inside the cone
    pt_sum = 0.
    pt_sum_global = 0.
    for track in fj_particles:
        if self.utils.delta_R(track, eta, phi) < jetR:
            pt_sum += track.pt()
        pt_sum_global += track.pt()
            
    if before_CS:
        delta_pt = pt_sum - rho * np.pi * jetR * jetR
        getattr(self, 'hDeltaPt_RC_beforeCS_R{}_Rmax{}'.format(jetR, R_max)).Fill(delta_pt)
    else:
        delta_pt = pt_sum
        getattr(self, 'hDeltaPt_RC_afterCS_R{}_Rmax{}'.format(jetR, R_max)).Fill(delta_pt)
        
    # Fill mean pt
    if before_CS and self.fill_R_indep_hists and self.fill_Rmax_indep_hists:
      N_tracks = len(fj_particles)
      mean_pt = pt_sum_global/N_tracks
      getattr(self, 'hN_MeanPt').Fill(N_tracks, mean_pt)

  #---------------------------------------------------------------
  # Fill truth jet histograms
  #---------------------------------------------------------------
  def fill_truth_before_matching(self, jet, jetR):
    
    jet_pt = jet.pt()
    for constituent in jet.constituents():
      z = constituent.pt() / jet.pt()
      getattr(self, 'hZ_Truth_R{}'.format(jetR)).Fill(jet.pt(), z)
          
    # Fill 2D histogram of truth (pt, obs)
    hname = 'h_{{}}_JetPt_Truth_R{}_{{}}'.format(jetR)
    self.fill_unmatched_jet_histograms(jet, jetR, hname)

  #---------------------------------------------------------------
  # Fill det jet histograms
  #---------------------------------------------------------------
  def fill_det_before_matching(self, jet, jetR, R_max, rho_bge = 0):
    
    if self.is_pp or self.fill_Rmax_indep_hists:
      jet_pt = jet.pt()
      if self.do_rho_subtraction:
        jet_pt = jet.pt()-rho_bge*jet.area()
      for constituent in jet.constituents():
        z = constituent.pt() / jet_pt
        getattr(self, 'hZ_Det_R{}'.format(jetR)).Fill(jet_pt, z)
      
    # for const in jet.constituents():
    #   if const.perp()>0.15:
    #     print('index',const.user_index(),'pt',const.perp())
    # print('fill det hist')
    
    # Fill groomed histograms
    if self.thermal_model:
      hname = 'h_{{}}_JetPt_R{}_{{}}_Rmax{}'.format(jetR, R_max)
      self.fill_unmatched_jet_histograms(jet, jetR, hname, rho_bge)

    if self.is_pp:
      hname = 'h_{{}}_JetPt_R{}_{{}}'.format(jetR)
      self.fill_unmatched_jet_histograms(jet, jetR, hname, rho_bge)

    if self.do_rho_subtraction:
      hname = 'h_{{}}_JetPt_R{}_{{}}'.format(jetR)
      self.fill_unmatched_jet_histograms(jet, jetR, hname, rho_bge)
  
  #---------------------------------------------------------------
  # This function is called once for each jet
  #---------------------------------------------------------------
  def fill_unmatched_jet_histograms(self, jet, jetR, hname, rho_bge = 0):

    # Loop through each jet subconfiguration (i.e. subobservable / grooming setting)
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
        
      if self.do_rho_subtraction and rho_bge > 0:
        jet_pt = jet.perp()-rho_bge*jet.area() # use subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      else:
        jet_pt = jet.perp()

      # Call user function to fill histograms
      self.fill_observable_histograms(hname, jet, jet_groomed_lund, jetR, obs_setting,
                                      grooming_setting, obs_label, jet_pt)
  
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

  def copy_parts(self, parts):
    # don't need to re-init every part for a deep copy
    parts_copied = fj.vectorPJ()
    for part in parts:
      # user_index_new = part.user_index()
      # part_new = fj.PseudoJet(part.px(), part.py(), part.pz(), part.E())
      # part_new.set_user_index(user_index_new)
      parts_copied.push_back(part)
    
    return parts_copied

  #---------------------------------------------------------------
  # Loop through jets and call user function to fill matched
  # histos if both det and truth jets are unique match.
  #---------------------------------------------------------------
  def fill_jet_matches(self, jet_det, jetR, R_max, fj_particles_det_holes, fj_particles_truth_holes, rho_bge = 0, fj_particles_det_cones = None, fj_particles_truth_cones = None):
  
    # Set suffix for filling histograms
    if R_max:
      suffix = '_Rmax{}'.format(R_max)
    else:
      suffix = ''
    
    # Get matched truth jet
    if jet_det.has_user_info():
      jet_truth = jet_det.python_info().match
      if self.do_rho_subtraction and rho_bge > 0:
        jet_det_pt = jet_det.perp()-rho_bge*jet_det.area() # use subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
      else:
        jet_det_pt = jet_det.perp()

      if jet_truth:

        # # debug
        # print('debug8--jet det', jet_det_pt, 'size', len(jet_det.constituents()))
        # print('debug8--jet_truth', jet_truth.pt(), 'size', len(jet_truth.constituents()))
        
        jet_pt_det_ungroomed = jet_det_pt
        jet_pt_truth_ungroomed = jet_truth.pt()
        JES = (jet_pt_det_ungroomed - jet_pt_truth_ungroomed) / jet_pt_truth_ungroomed
        getattr(self, 'hJES_R{}{}'.format(jetR, suffix)).Fill(jet_pt_truth_ungroomed, JES)
        
        # If Pb-Pb case, we need to keep jet_det, jet_truth, jet_pp_det
        jet_pp_det = None
        if not self.is_pp:
        
          # Get pp-det jet
          jet_pp_det = jet_truth.python_info().match
            
          # Fill delta-pt histogram
          if jet_pp_det:
            jet_pp_det_pt = jet_pp_det.pt()
            delta_pt = (jet_pt_det_ungroomed - jet_pp_det_pt)
            getattr(self, 'hDeltaPt_emb_R{}_Rmax{}'.format(jetR, R_max)).Fill(jet_pt_truth_ungroomed, delta_pt)
            
        # Loop through each jet subconfiguration (i.e. subobservable / grooming setting)
        observable = self.observable_list[0]
        for i in range(len(self.obs_settings[observable])):
        
          obs_setting = self.obs_settings[observable][i]
          grooming_setting = self.obs_grooming_settings[observable][i]
          obs_label = self.utils.obs_label(obs_setting, grooming_setting)
          
          if self.debug_level > 3:
            print('obs_label: {}'.format(obs_label))
          
          # Groom jets, if applicable
          if grooming_setting:
                    
            # Groom det jet
            gshop_det = fjcontrib.GroomerShop(jet_det, jetR, self.reclustering_algorithm)
            jet_det_groomed_lund = self.utils.groom(gshop_det, grooming_setting, jetR)
            if not jet_det_groomed_lund:
              continue

            # Groom truth jet
            gshop_truth = fjcontrib.GroomerShop(jet_truth, jetR, self.reclustering_algorithm)
            jet_truth_groomed_lund = self.utils.groom(gshop_truth, grooming_setting, jetR)
            if not jet_truth_groomed_lund:
              continue
              
          else:
          
            jet_det_groomed_lund = None
            jet_truth_groomed_lund = None
            
          # If jetscape, pass the list of holes within R of the jet to the user
          holes_in_det_jet = None
          holes_in_truth_jet = None
          if self.jetscape:
            holes_in_det_jet = [hadron for hadron in fj_particles_det_holes if jet_det.delta_R(hadron) < jetR]
            holes_in_truth_jet = [hadron for hadron in fj_particles_truth_holes if jet_truth.delta_R(hadron) < jetR]
            
            # Get the corrected jet pt by subtracting the negative recoils within R
            for hadron in holes_in_det_jet:
                jet_pt_det_ungroomed -= hadron.pt()
                
            for hadron in holes_in_truth_jet:
                jet_pt_truth_ungroomed -= hadron.pt()

          # Call user function to fill histos
          self.fill_matched_jet_histograms(jet_det, jet_det_groomed_lund, jet_truth,
                               jet_truth_groomed_lund, jet_pp_det, jetR,
                               obs_setting, grooming_setting, obs_label,
                               jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                               R_max, suffix, holes_in_det_jet=holes_in_det_jet,
                               holes_in_truth_jet=holes_in_truth_jet, cone_parts_in_det_jet=None, cone_parts_in_truth_jet=None, cone_R=0)

          # If check cone, pass the list of cone particles
          if self.do_jetcone:
            for jetcone_R in self.jetcone_R_list:
              
              cone_parts_in_det_jet = self.find_parts_around_jet(fj_particles_det_cones, jet_det, jetcone_R)
              cone_parts_in_truth_jet = self.find_parts_around_jet(fj_particles_truth_cones, jet_truth, jetcone_R)

              # Call user function to fill histos
              self.fill_matched_jet_histograms(jet_det, jet_det_groomed_lund, jet_truth,
                                 jet_truth_groomed_lund, jet_pp_det, jetR,
                                 obs_setting, grooming_setting, obs_label,
                                 jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                 R_max, suffix, holes_in_det_jet=holes_in_det_jet,
                                 holes_in_truth_jet=holes_in_truth_jet, cone_parts_in_det_jet=cone_parts_in_det_jet, cone_parts_in_truth_jet=cone_parts_in_truth_jet, cone_R=jetcone_R)

          # If check perpendicular cone, pass the list of perp cone particles (NB: re-label particle origins according to whether they are from perp or jet instead of pythia or PbPb. Need to make sure this part does not interfere with the ss, sb, bb classification in standard embedding procedures)
          # NO perpcone for the jetcone definition yet, just for the AK jets
          if self.do_perpcone:

            perp_jet1 = fj.PseudoJet()
            perp_jet1.reset_PtYPhiM(jet_det.pt(), jet_det.rapidity(), jet_det.phi() + np.pi/2, jet_det.m())
            perp_jet2 = fj.PseudoJet()
            perp_jet2.reset_PtYPhiM(jet_det.pt(), jet_det.rapidity(), jet_det.phi() - np.pi/2, jet_det.m())

            # The current implementation only does perpcone for the standard AK jets. No bigger cones
            perpcone_R = jetR
            constituents = jet_det.constituents()
            parts_in_jet = self.copy_parts(constituents)

            print('****************************')

            # check if the index is unmodified in the oringal consitituents
            for i, part in enumerate(constituents):
              if part.user_index()<0:
                print('Original user index for constituents (i, pt, eta, phi)',i,part.pt(),part.eta(),part.phi(),'should be <0:',part.user_index())

            # NB: a deep copy of fj_particles_det_cones are made before re-labeling the particle user_index and assembling the perp cone parts
            parts_in_perpcone1 = self.find_parts_around_jet(fj_particles_det_cones, perp_jet1, perpcone_R)
            parts_in_perpcone1 = self.rotate_parts(parts_in_perpcone1, -np.pi/2)
              
            parts_in_perpcone2 = self.find_parts_around_jet(fj_particles_det_cones, perp_jet2, perpcone_R)
            parts_in_perpcone2 = self.rotate_parts(parts_in_perpcone2, +np.pi/2)
            
            parts_in_cone1 = fj.vectorPJ()
            for i, part in enumerate(parts_in_jet):
              if part.user_index()<0:
                print('change user index for particle (i, pt, eta, phi)',i,part.pt(),part.eta(),part.phi(),'from',part.user_index(),'to 1')
              part.set_user_index(1)
              parts_in_cone1.append(part)
            for i, part in enumerate(parts_in_perpcone1):
              # if part.user_index()>0:
              #   print('change user index for particle (i, pt, eta, phi)',i,part.pt(),part.eta(),part.phi(),'from',part.user_index(),'to -1')
              part.set_user_index(-99)
              parts_in_cone1.append(part)

            # # check if the index is updated in parts_in_cone1
            # for i, part in enumerate(parts_in_cone1):
            #   if part.user_index()==-1:
            #     print('New user index for particle (i, pt, eta, phi)',i,part.pt(),part.eta(),part.phi(),'should be -1:',part.user_index())

            # check if the index is unmodified in the oringal consitituents
            for i, part in enumerate(constituents):
              if part.user_index()<0:
                print('Previous user index for constituents (i, pt, eta, phi)',i,part.pt(),part.eta(),part.phi(),'should be <0:',part.user_index())

            # # check if the index is unmodified in the oringal particles (before jet clustering)
            # for i, part in enumerate(fj_particles_det_cones):
            #   if part.user_index()>0:
            #     print('Previous user index for original particle (i, pt, eta, phi)',i,part.pt(),part.eta(),part.phi(),'should be >0:',part.user_index())
            
            parts_in_cone2 = fj.vectorPJ()
            for part in parts_in_jet:
              part.set_user_index(1)
              parts_in_cone2.append(part)
            for part in parts_in_perpcone2:
              part.set_user_index(-99)
              parts_in_cone2.append(part)
              
            cone_parts_in_det_jet = parts_in_cone1
            cone_parts_in_truth_jet = None

            # Call user function to fill histos
            self.fill_matched_jet_histograms(jet_det, jet_det_groomed_lund, jet_truth,
                                 jet_truth_groomed_lund, jet_pp_det, jetR,
                                 obs_setting, grooming_setting, obs_label,
                                 jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                 R_max, suffix, holes_in_det_jet=holes_in_det_jet,
                                 holes_in_truth_jet=holes_in_truth_jet, cone_parts_in_det_jet=cone_parts_in_det_jet, cone_parts_in_truth_jet=cone_parts_in_truth_jet, cone_R=0) # jetR used later so cone_R jet set to 0

            cone_parts_in_det_jet = parts_in_cone2
            cone_parts_in_truth_jet = None

            # Call user function to fill histos
            self.fill_matched_jet_histograms(jet_det, jet_det_groomed_lund, jet_truth,
                                 jet_truth_groomed_lund, jet_pp_det, jetR,
                                 obs_setting, grooming_setting, obs_label,
                                 jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                 R_max, suffix, holes_in_det_jet=holes_in_det_jet,
                                 holes_in_truth_jet=holes_in_truth_jet, cone_parts_in_det_jet=cone_parts_in_det_jet, cone_parts_in_truth_jet=cone_parts_in_truth_jet, cone_R=0) # jetR used later so cone_R jet set to 0

  #---------------------------------------------------------------
  # Fill response histograms -- common utility function
  #---------------------------------------------------------------
  def fill_response(self, observable, jetR, jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                    obs_det, obs_truth, obs_label, R_max, prong_match = False):

    if self.fill_RM_histograms:
      x = ([jet_pt_det_ungroomed, jet_pt_truth_ungroomed, obs_det, obs_truth])
      x_array = array('d', x)
      name = 'hResponse_JetPt_{}_R{}_{}'.format(observable, jetR, obs_label)
      if not self.is_pp:
        name += '_Rmax{}'.format(R_max)
      getattr(self, name).Fill(x_array)
      
    if obs_truth > 1e-5:
      obs_resolution = (obs_det - obs_truth) / obs_truth
      name = 'hResidual_JetPt_{}_R{}_{}'.format(observable, jetR, obs_label)
      if not self.is_pp:
        name += '_Rmax{}'.format(R_max)
      getattr(self, name).Fill(jet_pt_truth_ungroomed, obs_truth, obs_resolution)
    
    # Fill prong-matched response
    if not self.is_pp and R_max == self.main_R_max:
      if prong_match:
      
        name = 'hResponse_JetPt_{}_R{}_{}_Rmax{}_matched'.format(observable, jetR, obs_label, R_max)
        getattr(self, name).Fill(x_array)
        
        if obs_truth > 1e-5:
          name = 'hResidual_JetPt_{}_R{}_{}_Rmax{}_matched'.format(observable, jetR, obs_label, R_max)
          getattr(self, name).Fill(jet_pt_truth_ungroomed, obs_truth, obs_resolution)

  #---------------------------------------------------------------
  # This function is called once for each jetR
  # You must implement this
  #---------------------------------------------------------------
  def initialize_user_output_objects_R(self, jetR):
      
    raise NotImplementedError('You must implement initialize_user_output_objects_R()!')

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # You must implement this
  #---------------------------------------------------------------
  def fill_observable_histograms(self, hname, jet, jet_groomed_lund, jetR, obs_setting,
                                 grooming_setting, obs_label, jet_pt_ungroomed):

    raise NotImplementedError('You must implement fill_observable_histograms()!')

  #---------------------------------------------------------------
  # This function is called once for each matched jet subconfiguration
  # You must implement this
  #---------------------------------------------------------------
  def fill_matched_jet_histograms(self, jet_det, jet_det_groomed_lund, jet_truth,
                                  jet_truth_groomed_lund, jet_pp_det, jetR,
                                  obs_setting, grooming_setting, obs_label,
                                  jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                  R_max, suffix,
                                  **kwargs):

    raise NotImplementedError('You must implement fill_matched_jet_histograms()!')
