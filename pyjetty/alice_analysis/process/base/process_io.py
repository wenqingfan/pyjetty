#!/usr/bin/env python3

"""
  Analysis IO class for jet analysis with track dataframe.
  Each instance of the class handles the IO of a *single* track tree.
  
  Authors: James Mulligan
           Mateusz Ploskon
           Ezra Lesser
"""

from __future__ import print_function

import os   # for creating file on output
import sys

# Data analysis and plotting
import uproot
import pandas
import numpy as np
import yaml

# Fastjet via python (from external library fjpydev)
import fastjet as fj
import fjext

# Base class
from pyjetty.alice_analysis.process.base import common_base

################################################################
class ProcessIO(common_base.CommonBase):
  
  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', tree_dir='PWGHF_TreeCreator',
               track_tree_name='tree_Particle', event_tree_name='tree_event_char',
               output_dir='', is_pp=True, min_cent=0., max_cent=10.,
               use_ev_id_ext=True, is_jetscape=False, holes=False,
               event_plane_range=None, skip_event_tree=False, is_ENC=False, is_det_level=False, **kwargs):
    super(ProcessIO, self).__init__(**kwargs)
    self.input_file = input_file
    self.output_dir = output_dir
    self.tree_dir = tree_dir
    if len(tree_dir) and tree_dir[-1] != '/':
      self.tree_dir += '/'
    self.track_tree_name = track_tree_name
    self.event_tree_name = event_tree_name
    self.is_pp = is_pp
    self.use_ev_id_ext = use_ev_id_ext
    self.is_jetscape = is_jetscape
    self.holes = holes
    self.event_plane_range = event_plane_range
    self.skip_event_tree = skip_event_tree
    self.is_ENC = is_ENC
    self.is_det_level = is_det_level
    if len(output_dir) and output_dir[-1] != '/':
      self.output_dir += '/'
    self.reset_dataframes()
    
    # Set the combination of fields that give a unique event id
    self.unique_identifier =  ['run_number', 'ev_id']
    if self.use_ev_id_ext:
      if self.is_pp:
        self.unique_identifier += ['ev_id_ext']
      else:
        pass # somehow for PbPb there is no 'ev_id_ext'
      
    # Set relevant columns of event tree
    self.event_columns = self.unique_identifier + ['z_vtx_reco', 'is_ev_rej']
    if not self.is_pp:
      self.event_columns += ['centrality']
      self.min_centrality = min_cent
      self.max_centrality = max_cent
    if is_jetscape:
      self.event_columns += ['event_plane_angle']
    
    # Set relevant columns of track tree
    self.track_columns = self.unique_identifier + ['ParticlePt', 'ParticleEta', 'ParticlePhi']
    if is_jetscape:
        self.track_columns += ['status']
    if is_ENC:
        if is_det_level:
          self.track_columns += ['ParticleMCIndex']
        else:
          self.track_columns += ['ParticlePID']
    
    #print(self)
    
  #---------------------------------------------------------------
  # Clear dataframes
  #---------------------------------------------------------------
  def reset_dataframes(self):
    self.event_df_orig = None
    self.track_df = None
  
  #---------------------------------------------------------------
  # Convert ROOT TTree to SeriesGroupBy object of fastjet particles per event.
  # Optionally, define the mass assumption used in the jet reconstruction;
  #             remove a certain random fraction of tracks;
  #             randomly assign proton and kaon mass to some tracks
  #---------------------------------------------------------------
  def load_data(self, m=0.1396, reject_tracks_fraction=0., reject_tracks_config=None, offset_indices=False,
                group_by_evid=True, random_mass=False, min_pt=0.):
    
    self.reject_tracks_fraction = reject_tracks_fraction
    self.reset_dataframes()

    print('Convert ROOT trees to pandas dataframes...')
    print('    track_tree_name = {}'.format(self.track_tree_name))

    self.track_df = self.load_dataframe()

    # Check if pT-dependent track keeping is requested
    if reject_tracks_config:
      with open(reject_tracks_config, 'r') as f:
        config = yaml.safe_load(f)
      
      pt_bins = config.get('ptBinning', [0., 999.])
      keep_fractions = config.get('keepFraction', [1.0])  # Use 'keepFraction' as keep_fractions
      print('Using the following input for tracking efficiency uncertainty')
      print('pt binning:',pt_bins)
      print('fraction of tracks to keep:',keep_fractions)
      
      # Ensure keep_fractions matches the length of pt_bins - 1
      if len(keep_fractions) != len(pt_bins) - 1:
        raise ValueError(f"Length of keepFraction ({len(keep_fractions)}) "
                         f"must be one less than ptBinning ({len(pt_bins)})")
      
      # Create a copy of the track dataframe to modify
      df_to_drop = self.track_df.copy()
      
      # Track keeping based on pT bins
      df_to_drop['bin_index'] = pandas.cut(df_to_drop['ParticlePt'], 
                                            bins=pt_bins, 
                                            labels=False, 
                                            include_lowest=True)
      
      # Tracks to keep for each pT bin
      tracks_to_keep = []
      for bin_index, keep_fraction in enumerate(keep_fractions):
        # Get tracks in this pT bin
        bin_tracks = df_to_drop[df_to_drop['bin_index'] == bin_index]
        
        # Number of tracks to keep
        n_keep = int(len(bin_tracks) * keep_fraction)
        
        if n_keep > 0:
          # Randomly select tracks to keep
          np.random.seed()
          indices_to_keep = np.random.choice(bin_tracks.index, n_keep, replace=False)
          tracks_to_keep.extend(indices_to_keep)
      
      # Keep only selected tracks
      if tracks_to_keep:
        print(f'    Keeping {len(tracks_to_keep)} of {len(self.track_df.index)} tracks with pT-dependent selection')
        self.track_df = self.track_df.loc[tracks_to_keep]
    
    # Original track rejection method (kept for backwards compatibility)
    elif self.reject_tracks_fraction > 1e-3:
      n_remove = int(reject_tracks_fraction * len(self.track_df.index))
      print('    Removing {} of {} tracks from {}'.format(
        n_remove, len(self.track_df.index), self.track_tree_name))
      np.random.seed()
      indices_remove = np.random.choice(self.track_df.index, n_remove, replace=False)
      self.track_df.drop(indices_remove, inplace=True)

    if random_mass:
      print('    \033[93mRandomly assigning proton and kaon mass to some tracks.\033[0m') 

    df_fjparticles = self.group_fjparticles(m, offset_indices, group_by_evid, random_mass, min_pt=min_pt)

    return df_fjparticles
  
  #---------------------------------------------------------------
  # Convert ROOT TTree to pandas dataframe
  # Return merged track+event dataframe from a given input file
  # Returned dataframe has one row per jet constituent:
  #     run_number, ev_id, ParticlePt, ParticleEta, ParticlePhi
  #---------------------------------------------------------------
  def load_dataframe(self):

    # Load event tree into dataframe
    if not self.skip_event_tree:
      event_tree = None
      event_df = None
      event_tree_name = self.tree_dir + self.event_tree_name
      with uproot.open(self.input_file)[event_tree_name] as event_tree:
        if not event_tree:
          raise ValueError("Tree %s not found in file %s" % (event_tree_name, self.input_file))
        self.event_df_orig = uproot.concatenate(event_tree, self.event_columns, library="pd")
    
      # Check if there are duplicated event ids
      #print(self.event_df_orig)
      #d = self.event_df_orig.duplicated(self.unique_identifier, keep=False)
      #print(self.event_df_orig[d])
      n_duplicates = sum(self.event_df_orig.duplicated(self.unique_identifier))
      if n_duplicates > 0:
        raise ValueError(
          "There appear to be %i duplicate events in the event dataframe" % n_duplicates)
      
      # Apply event selection
      self.event_df_orig.reset_index(drop=True)
      if self.is_pp:
        event_criteria = 'is_ev_rej == 0'
      else:
        event_criteria = 'is_ev_rej == 0 and centrality > @self.min_centrality and centrality < @self.max_centrality'
      if self.event_plane_range:
        event_criteria += ' and event_plane_angle > @self.event_plane_range[0] and event_plane_angle < @self.event_plane_range[1]'
      event_df = self.event_df_orig.query(event_criteria)
      event_df.reset_index(drop=True)

    # Load track tree into dataframe
    track_tree = None
    track_df_orig = None
    track_tree_name = self.tree_dir + self.track_tree_name
    with uproot.open(self.input_file)[track_tree_name] as track_tree:
      if not track_tree:
        raise ValueError("Tree %s not found in file %s" % (track_tree_name, self.input_file))
      track_df_orig = uproot.concatenate(track_tree, self.track_columns, library="pd")
    
    # Apply hole selection, in case of jetscape
    if self.is_jetscape:
        if self.holes:
            track_criteria = 'status == -1'
        else:
            track_criteria = 'status == 0'
        track_df_orig = track_df_orig.query(track_criteria)
        track_df_orig.reset_index(drop=True)
    
    # Check if there are duplicated tracks
    #print(track_df_orig)
    #d = track_df_orig.duplicated(self.track_columns, keep=False)
    #print(track_df_orig[d])
    n_duplicates = sum(track_df_orig.duplicated(self.track_columns))
    if n_duplicates > 0:
      raise ValueError(
        "There appear to be %i duplicate particles in the track dataframe" % n_duplicates)

    # Merge event info into track tree
    if self.skip_event_tree:
      self.track_df = track_df_orig
    else:
      self.track_df = pandas.merge(track_df_orig, event_df, on=self.unique_identifier)
    
    # Check if there are duplicated tracks in the merge dataframe
    #print(self.track_df)
    #d = self.track_df.duplicated(self.track_columns, keep=False)
    #print(self.track_df[d])
    n_duplicates = sum(self.track_df.duplicated(self.track_columns))
    if n_duplicates > 0:
      sys.exit('ERROR: There appear to be {} duplicate particles in the merged dataframe'.format(n_duplicates))
      
    return self.track_df

  #---------------------------------------------------------------
  # Opposite operation as load_dataframe above. Takes a dataframe
  # with the same formatting and saves to class's output_file.
  # histograms is list of tuples: [ ("title", np.histogram), ... ]
  #---------------------------------------------------------------
  def save_dataframe(self, filename, df, df_true=False, histograms=[], is_jetscape=False, is_ENC=False):

    # Create output directory if it does not already exist
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Open output directory and (re)create rootfile
    with uproot.recreate(self.output_dir + filename) as f:

      branchdict = {"run_number": int, "ev_id": int, "ParticlePt": float,
                      "ParticleEta": float, "ParticlePhi": float}
      if is_jetscape:
        branchdict_true["status"] = int
        branchdict["status"] = int

      if is_ENC:
        branchdict_true["ParticlePID"] = int
        branchdict["ParticleMCIndex"] = int

      if df_true:
        # Create tree with truth particle info
        title = 'tree_Particle_gen'
        print("Length of truth track tree: %i" % len(self.track_df))
        f.mktree(name=title, branch_types=branchdict_true, title=title)
        if is_jetscape:
            f[title].extend( { "run_number": self.track_df["run_number"],
                               "ev_id": self.track_df["ev_id"],
                               "ParticlePt": self.track_df["ParticlePt"],
                               "ParticleEta": self.track_df["ParticleEta"],
                               "ParticlePhi": self.track_df["ParticlePhi"],
                               "status": self.track_df["status"] } )
        elif is_ENC:
          f[title].extend( { "run_number": self.track_df["run_number"],
                               "ev_id": self.track_df["ev_id"],
                               "ParticlePt": self.track_df["ParticlePt"],
                               "ParticleEta": self.track_df["ParticleEta"],
                               "ParticlePhi": self.track_df["ParticlePhi"],
                               "ParticlePID": self.track_df["ParticlePID"] } ) # to get charge info
        else:
            f[title].extend( { "run_number": self.track_df["run_number"],
                               "ev_id": self.track_df["ev_id"],
                               "ParticlePt": self.track_df["ParticlePt"],
                               "ParticleEta": self.track_df["ParticleEta"],
                               "ParticlePhi": self.track_df["ParticlePhi"] } )

      # Create tree with detector-level particle info
      title = 'tree_Particle'
      print("Length of detector-level track tree: %i" % len(df))
      f.mktree(name=title, branch_types=branchdict, title=title)
      if is_jetscape:
        f[title].extend( { "run_number": df["run_number"],
                           "ev_id": df["ev_id"],
                           "ParticlePt": df["ParticlePt"],
                           "ParticleEta": df["ParticleEta"],
                           "ParticlePhi": df["ParticlePhi"],
                           "status": df["status"] } )
      elif is_ENC:
        f[title].extend( { "run_number": df["run_number"],
                               "ev_id": df["ev_id"],
                               "ParticlePt": df["ParticlePt"],
                               "ParticleEta": df["ParticleEta"],
                               "ParticlePhi": df["ParticlePhi"],
                               "ParticleMCIndex": df["ParticleMCIndex"] } ) # associated MC particle index
      else:
        f[title].extend( { "run_number": df["run_number"],
                           "ev_id": df["ev_id"],
                           "ParticlePt": df["ParticlePt"],
                           "ParticleEta": df["ParticleEta"],
                           "ParticlePhi": df["ParticlePhi"] } )

      # Create tree with event char
      title = self.event_tree_name
      branchdict = {"is_ev_rej": int, "run_number": int, "ev_id": int, "z_vtx_reco": float}
      if is_jetscape:
        branchdict["event_plane_angle"] = float
      f.mktree(name=title, branch_types=branchdict, title=title)
      if is_jetscape:
        f[title].extend( {"is_ev_rej": self.event_df_orig["is_ev_rej"], 
                        "run_number": self.event_df_orig["run_number"], 
                        "ev_id": self.event_df_orig["ev_id"],
                        "z_vtx_reco": self.event_df_orig["z_vtx_reco"],
                        "event_plane_angle": self.event_df_orig["event_plane_angle"] } )
      else:
        f[title].extend( {"is_ev_rej": self.event_df_orig["is_ev_rej"], 
                        "run_number": self.event_df_orig["run_number"], 
                        "ev_id": self.event_df_orig["ev_id"],
                        "z_vtx_reco": self.event_df_orig["z_vtx_reco"] } )
        
      # Write hNevents histogram: number of accepted events at detector level
      f["hNevents"] = ( np.array([ 0, df["ev_id"].nunique() ]), np.array([ -0.5, 0.5, 1.5 ]) )

      # Write histograms to file too, if any are passed
      for title, h in histograms:
        f[title] = h

  #---------------------------------------------------------------
  # Transform the track dataframe into a SeriesGroupBy object
  # of fastjet particles per event.
  #---------------------------------------------------------------
  def group_fjparticles(self, m, offset_indices=False, group_by_evid=True, random_mass=False, min_pt=0.):

    print('is_ENC on?',self.is_ENC)
    print('is_det on?',self.is_det_level)
    print('debug',self.track_df)
    if group_by_evid:
      print("Transform the track dataframe into a series object of fastjet particles per event...")

      # (i) Group the track dataframe by event
      #     track_df_grouped is a DataFrameGroupBy object with one track dataframe per event
      track_df_grouped = None
      track_df_grouped = self.track_df.groupby(self.unique_identifier)
      print('debug2',type(track_df_grouped))
      print('debug2',track_df_grouped.aggregate(np.sum))
      # print('debug2',track_df_grouped.columns['ParticlePID'].values)
    
      # (ii) Transform the DataFrameGroupBy object to a SeriesGroupBy of fastjet particles
      df_fjparticles = None
      

      if self.is_ENC:
        df_fjparticles_orig = track_df_grouped.apply(
        self.get_fjparticles, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
        if self.is_det_level:
          df_fjparticles_aux = track_df_grouped.apply(
          self.get_particles_mc_index, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
          print('debug3',df_fjparticles)
          print('debug3 aux: mcid',df_fjparticles_aux)
          df_fjparticles = pandas.DataFrame({"fj_particle": df_fjparticles_orig, "ParticleMCIndex": df_fjparticles_aux})
        else:
          df_fjparticles_aux = track_df_grouped.apply(
          self.get_particles_pid, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
          print('debug3',df_fjparticles)
          print('debug3 aux: pid',df_fjparticles_aux)
          df_fjparticles = pandas.DataFrame({"fj_particle": df_fjparticles_orig, "ParticlePID": df_fjparticles_aux})
      else:
        df_fjparticles = track_df_grouped.apply(
        self.get_fjparticles, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
      
      print('debug4, combined: ',df_fjparticles)
      
      # df_fjparticles = pandas.DataFrame({"fj_particle": track_df_grouped.apply(
      #   self.get_fjparticles, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt), "ParticleMCIndex": track_df_grouped["ParticleMCIndex"]})
      
    
    else:
      print("Transform the track dataframe into a dataframe of fastjet particles per track...")

      # Transform into a DataFrame of fastjet particles
      # if it's for energy correlator analysis, add particle id and associated MC info for truth and det level input respectively
      df = self.track_df
      if self.is_ENC:
        if self.is_det_level:
          df_fjparticles = pandas.DataFrame( 
            {"run_number": df["run_number"], "ev_id": df["ev_id"],
            "fj_particle": self.get_fjparticles(self.track_df, m, offset_indices, random_mass, min_pt=  min_pt), "ParticleMCIndex": df["ParticleMCIndex"]} )
        else:
          df_fjparticles = pandas.DataFrame( 
            {"run_number": df["run_number"], "ev_id": df["ev_id"],
            "fj_particle": self.get_fjparticles(self.track_df, m, offset_indices, random_mass, min_pt=  min_pt), "ParticlePID": df["ParticlePID"]} )
      else:
        df_fjparticles = pandas.DataFrame( 
          {"run_number": df["run_number"], "ev_id": df["ev_id"],
          "fj_particle": self.get_fjparticles(self.track_df, m, offset_indices, random_mass, min_pt=  min_pt)} )

    return df_fjparticles

  #---------------------------------------------------------------
  # Return fastjet:PseudoJets from a given track dataframe
  #---------------------------------------------------------------
  def get_fjparticles(self, df_tracks, m, offset_indices=False, random_mass=False, min_pt=0.):
    
    # If offset_indices is true, then offset the user_index by a large negative value
    user_index_offset = 0
    if offset_indices:
        user_index_offset = int(-1e6)
        
    # Apply a pt cut
    df_tracks_accepted = df_tracks[df_tracks.ParticlePt > min_pt]

    # print('debug2',df_tracks_accepted)

    m_array = np.full((df_tracks_accepted['ParticlePt'].values.size), m)

    # Randomly assign K and p mass for systematic check
    if random_mass:
      rand_val = np.random.random((len(m_array)))
      K_mass = 0.4937     # GeV/c^2
      p_mass = 0.938272   # GeV/c^2
      # (p + pbar) / (pi+ + pi-) ~ 5.5%
      # (K+ + K-) / (pi+ + pi-) ~ 13%
      # But these are numbers with respect to the final _unreplaced_ pions, so there is
      # an additional factor of 1/(1 + 5.5% + 13%) to get things right
      K_factor = 0.13; p_factor = 0.055
      K_prob = K_factor / (1 + K_factor + p_factor)
      p_prob = 1 - p_factor / (1 + K_factor + p_factor)   # 1- just to look at diff random vals
      m_array = np.where(rand_val < K_prob, K_mass, m_array)
      m_array = np.where(rand_val > p_prob, p_mass, m_array)

    # Use swig'd function to create a vector of fastjet::PseudoJets from numpy arrays of pt,eta,phi
    fj_particles = fjext.vectorize_pt_eta_phi_m(
      df_tracks_accepted['ParticlePt'].values, df_tracks_accepted['ParticleEta'].values,
      df_tracks_accepted['ParticlePhi'].values, m_array, user_index_offset)

    return fj_particles
    # if self.is_ENC:
    #   if self.is_det_level:
    #     return fj_particles, df_tracks_accepted['ParticleMCIndex'].values
    #   else:
    #     return fj_particles, df_tracks_accepted['ParticlePID'].values
    # else:
    #   return fj_particles

  #---------------------------------------------------------------
  # Return associated mc indices from a given track dataframe
  #---------------------------------------------------------------
  def get_particles_mc_index(self, df_tracks, m, offset_indices=False, random_mass=False, min_pt=0.):
    
    # If offset_indices is true, then offset the user_index by a large negative value
    user_index_offset = 0
    if offset_indices:
        user_index_offset = int(-1e6)
        
    # Apply a pt cut
    df_tracks_accepted = df_tracks[df_tracks.ParticlePt > min_pt]

    m_array = np.full((df_tracks_accepted['ParticlePt'].values.size), m)

    return df_tracks_accepted['ParticleMCIndex'].values

  #---------------------------------------------------------------
  # Return particle id from a given track dataframe
  #---------------------------------------------------------------
  def get_particles_pid(self, df_tracks, m, offset_indices=False, random_mass=False, min_pt=0.):
    
    # If offset_indices is true, then offset the user_index by a large negative value
    user_index_offset = 0
    if offset_indices:
        user_index_offset = int(-1e6)
        
    # Apply a pt cut
    df_tracks_accepted = df_tracks[df_tracks.ParticlePt > min_pt]

    m_array = np.full((df_tracks_accepted['ParticlePt'].values.size), m)

    return df_tracks_accepted['ParticlePID'].values
