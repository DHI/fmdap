// Created     : 2019-11-6 20:42:5
// DLL         : C:\Program Files (x86)\DHI\2020\bin\x64\pfs2004.dll
// Version     : 18.0.0.13265

[FemEngineSW]
   [DOMAIN]
      Touched = 1
      discretization = 2
      number_of_dimensions = 2
      number_of_meshes = 1
      file_name = |.\input\SW_local_DWF_MSL_coarse_v2.mesh|
      type_of_reordering = 1
      number_of_domains = 16
      coordinate_type = 'LONG/LAT'
      minimum_depth = 16.83972962137465
      datum_depth = 0.0
      vertical_mesh_type_overall = 1
      number_of_layers = 10
      z_sigma = -1378.329124307562
      vertical_mesh_type = 1
      layer_thickness = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
      sigma_c = 0.1
      theta = 2.0
      b = 0.0
      number_of_layers_zlevel = 10
      vertical_mesh_type_zlevel = 1
      constant_layer_thickness_zlevel = 137.8329124307562
      variable_layer_thickness_zlevel = 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562
      type_of_bathymetry_adjustment = 2
      minimum_layer_thickness_zlevel = 1.378329124307562
      type_of_mesh = 0
      type_of_gauss = 3
      [BOUNDARY_NAMES]
         Touched = 0
         MzSEPfsListItemCount = 2
         [CODE_3]
            Touched = 0
            name = 'Code 3'
         EndSect  // CODE_3

         [CODE_5]
            Touched = 0
            name = 'Code 5'
         EndSect  // CODE_5

      EndSect  // BOUNDARY_NAMES

   EndSect  // DOMAIN

   [TIME]
      Touched = 1
      start_time = 2017, 10, 27, 0, 0, 0
      time_step_interval = 600.0
      number_of_time_steps = 30
   EndSect  // TIME

   [MODULE_SELECTION]
      Touched = 1
      mode_of_hydrodynamic_module = 0
      hydrodynamic_features = 1
      fluid_property = 1
      mode_of_spectral_wave_module = 2
      mode_of_transport_module = 0
      mode_of_mud_transport_module = 0
      mode_of_eco_lab_module = 0
      mode_of_sand_transport_module = 0
      mode_of_particle_tracking_module = 0
      mode_of_oil_spill_module = 0
      mode_of_shoreline_module = 0
	  mode_of_data_assimilation_module = 2
   EndSect  // MODULE_SELECTION

   [SPECTRAL_WAVE_MODULE]
      mode = 2
      [SPACE]
         number_of_mesh_geometry = 1
      EndSect  // SPACE

      [EQUATION]
         Touched = 1
         formulation = 2
         time_formulation = 2
         JONSWAP_factor_1 = 0.92
         JONSWAP_factor_2 = 0.83
      EndSect  // EQUATION

      [TIME]
         Touched = 1
         start_time_step = 0
         time_step_factor = 1
         time_step_factor_AD = 1
      EndSect  // TIME

      [SPECTRAL]
         Touched = 1
         type_of_frequency_discretization = 2
         number_of_frequencies = 25
         minimum_frequency = 0.055
         frequency_interval = 0.02
         frequency_factor = 1.1
         type_of_directional_discretization = 1
         number_of_directions = 16
         minimum_direction = 0.0
         maximum_direction = 180.0
         separation_of_wind_sea_and_swell = 0
         threshold_frequency = 0.125
         maximum_threshold_frequency = 0.5959088268863615
      EndSect  // SPECTRAL

      [SOLUTION_TECHNIQUE]
         Touched = 1
         error_level = 0
         maximum_number_of_errors = 200
         minimum_period = 0.1
         maximum_period = 25.0
         initial_period = 8.0
         scheme_of_space_discretization_geographical = 1
         scheme_of_space_discretization_direction = 1
         scheme_of_space_discretization_frequency = 1
         method = 1
         number_of_iterations = 500
         tolerance1 = 1e-06
         tolerance2 = 0.001
         relaxation_factor = 0.1
         number_of_levels_in_transport_calc = 32
         number_of_steps_in_source_calc = 1
         maximum_CFL_number = 1.0
         dt_min = 0.01
         dt_max = 600.0
         type_overall = 0
         file_name_overall = 'convergence_overall.dfs0'
         input_format = 1
         coordinate_type = ''
         input_file_name = ||
         number_of_points = 0
         type_domain = 0
         file_name_domain = 'convergence_domain.dfsu'
         output_frequency = 5
      EndSect  // SOLUTION_TECHNIQUE

      [DEPTH]
         Touched = 1
         type = 1
         minimum_depth = 0.01
         format = 3
         soft_time_interval = 0.0
         constant_level = 0.0
         file_name = |.\input\HD_1hr.dfsu|
         item_number = 1
         item_name = 'Surface elevation'
      EndSect  // DEPTH

      [CURRENT]
         Touched = 1
         type = 1
         type_blocking = 1
         factor_blocking = 0.1
         format = 3
         soft_time_interval = 0.0
         constant_x_velocity = 0.0
         constant_y_velocity = 0.0
         file_name = |.\input\HD_1hr.dfsu|
         item_number_for_x_velocity = 2
         item_number_for_y_velocity = 3
         item_name_for_x_velocity = 'Current velocity, U'
         item_name_for_y_velocity = 'Current velocity, V'
      EndSect  // CURRENT

      [WIND]
         Touched = 1
         type = 1
         format = 3
         constant_speed = 0.0
         constant_direction = 0.0
         file_name = |.\input\Wind_1hr.dfsu|
         item_number_for_speed = 1
         item_number_for_direction = 2
         item_name_for_speed = 'Wind speed'
         item_name_for_direction = 'Wind direction'
         soft_time_interval = 0.0
         formula = 1
         type_of_drag = 1
         linear_growth_coefficient = 0.0015
         type_of_air_sea_interaction = 0
         background_Charnock_parameter = 0.01
         Charnock_parameter = 0.02
         alpha_drag = 0.00063
         beta_drag = 6.600000000000001e-05
      EndSect  // WIND

      [ICE]
         Touched = 1
         type = 0
         format = 3
         c_cut_off = 0.33
         file_name = ||
         item_number = 1
         item_name = ''
      EndSect  // ICE

      [DIFFRACTION]
         Touched = 1
         type = 0
         minimum_delta = -0.75
         maximum_delta = 3.0
         type_of_smoothing = 1
         smoothing_factor = 1.0
         number_of_smoothing_steps = 1
      EndSect  // DIFFRACTION

      [TRANSFER]
         Touched = 1
         type = 1
         type_triad = 0
         alpha_EB = 0.25
      EndSect  // TRANSFER

      [WAVE_BREAKING]
         Touched = 1
         type = 1
         type_of_gamma = 1
         alpha = 1.0
         gamma_steepness = 1.0
         type_of_effect_on_frequency = 0
         type_of_roller = 0
         roller_propagation_factor = 1.0
         roller_dissipation_factor = 0.15
         roller_density = 1000.0
         [GAMMA]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.8
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // GAMMA

      EndSect  // WAVE_BREAKING

      [BOTTOM_FRICTION]
         Touched = 1
         type = 3
         constant_fc = 0.0
         type_of_effect_on_frequency = 1
         [FRICTION_COEFFICIENT]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.0077
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // FRICTION_COEFFICIENT

         [FRICTION_FACTOR]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.0212
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // FRICTION_FACTOR

         [NIKURADSE_ROUGHNESS]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.01
            file_name = |.\input\HKZN_mesh_v2_final_BF_map.dfsu|
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // NIKURADSE_ROUGHNESS

         [SAND_GRAIN_SIZE]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.00025
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // SAND_GRAIN_SIZE

      EndSect  // BOTTOM_FRICTION

      [WHITECAPPING]
         Touched = 1
         type = 1
         type_of_spectrum = 3
         mean_frequency_power = 1
         mean_wave_number_power = 1
         [dissipation_cdiss]
            Touched = 1
            type = 1
            format = 0
            constant_value = 2.1
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // dissipation_cdiss

         [dissipation_delta]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.337
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // dissipation_delta

      EndSect  // WHITECAPPING

      [STRUCTURES]
         type = 0
         input_format = 1
         coordinate_type = ''
         number_of_structures = 0
         input_file_name = ||
         [LINE_STRUCTURES]
            Touched = 1
            MzSEPfsListItemCount = 0
            output_of_link_data = 0
            file_name_section = 'line_section.xyz'
            number_of_structures = 0
         EndSect  // LINE_STRUCTURES

      EndSect  // STRUCTURES

      [INITIAL_CONDITIONS]
         Touched = 1
         type = 1
         type_additional = 1
         type_of_spectra = 1
         fetch = 100000.0
         max_peak_frequency = 0.4
         max_Phillips_constant = 0.0081
         shape_parameter_sigma_a = 0.07000000000000001
         shape_parameter_sigma_b = 0.09
         peakednes_parameter_gamma = 3.3
         file_name_m = ||
         item_number_m0 = 1
         item_number_m1 = 1
         item_name_m0 = ''
         item_name_m1 = ''
         file_name_A = ||
         item_number_A = 1
         item_name_A = ''
      EndSect  // INITIAL_CONDITIONS

      [BOUNDARY_CONDITIONS]
         Touched = 1
         MzSEPfsListItemCount = 2
         [CODE_1]
         EndSect  // CODE_1

         [CODE_3]
            Touched = 1
            type = 4
            format = 1
            constant_values = 1.0, 8.0, 270.0, 5.0, 0.1, 16.0, 270.0, 32.0
            file_name = |.\input\north_wave_params.dfs0|
            item_numbers = 1, 2, 3, 4, 1, 1, 1, 1
            item_names = 'Point 1: Sign. Wave Height', 'Point 1: Peak Wave Period', 'Point 1: Mean Wave Direction', 'Point 1: Dir. Stand. Deviation', 'Point 1: Sign. Wave Height', 'Point 1: Sign. Wave Height', 'Point 1: Sign. Wave Height', 'Point 1: Sign. Wave Height'
            type_of_soft_start = 1
            soft_time_interval = 0.0
            reference_values = 0.0, 8.0, 270.0, 5.0, 0.0, 16.0, 270.0, 32.0
            type_of_time_interpolation = 1, 1, 1, 1, 1, 1, 1, 1
            type_of_space_interpolation = 1
            code_cyclic = 0
            reflection_coefficient = 1.0
            type_of_frequency_spectrum = 2
            type_of_frequency_normalization = 1
            sigma_a = 0.07000000000000001
            sigma_b = 0.09
            gamma = 3.3
            type_of_directional_distribution = 1
            type_of_directional_normalization = 1
            type_of_frequency_spectrum_swell = 2
            type_of_frequency_normalization_swell = 1
            sigma_a_swell = 0.07000000000000001
            sigma_b_swell = 0.09
            gamma_swell = 5.0
            type_of_directional_distribution_swell = 1
            type_of_directional_normalization_swell = 1
         EndSect  // CODE_3

         [CODE_5]
            Touched = 1
            type = 4
            format = 1
            constant_values = 1.0, 8.0, 270.0, 5.0, 0.1, 16.0, 270.0, 32.0
            file_name = |.\input\south_wave_params.dfs0|
            item_numbers = 1, 2, 3, 4, 1, 1, 1, 1
            item_names = 'Point 1: Sign. Wave Height', 'Point 1: Peak Wave Period', 'Point 1: Mean Wave Direction', 'Point 1: Dir. Stand. Deviation', 'Point 1: Sign. Wave Height', 'Point 1: Sign. Wave Height', 'Point 1: Sign. Wave Height', 'Point 1: Sign. Wave Height'
            type_of_soft_start = 1
            soft_time_interval = 0.0
            reference_values = 0.0, 8.0, 270.0, 5.0, 0.0, 16.0, 270.0, 32.0
            type_of_time_interpolation = 1, 1, 1, 1, 1, 1, 1, 1
            type_of_space_interpolation = 1
            code_cyclic = 0
            reflection_coefficient = 1.0
            type_of_frequency_spectrum = 2
            type_of_frequency_normalization = 1
            sigma_a = 0.07000000000000001
            sigma_b = 0.09
            gamma = 3.3
            type_of_directional_distribution = 1
            type_of_directional_normalization = 1
            type_of_frequency_spectrum_swell = 2
            type_of_frequency_normalization_swell = 1
            sigma_a_swell = 0.07000000000000001
            sigma_b_swell = 0.09
            gamma_swell = 5.0
            type_of_directional_distribution_swell = 1
            type_of_directional_normalization_swell = 1
         EndSect  // CODE_5


      EndSect  // BOUNDARY_CONDITIONS

      [OUTPUTS]
         Touched = 1
         MzSEPfsListItemCount = 2
         number_of_outputs = 2
         [OUTPUT_1]
            Touched = 1
            include = 1
            title = 'Area'
            file_name = 'HKZN_local_2017_DutchCoast.dfsu'
            type = 1
            format = 2
            flood_and_dry = 2
            coordinate_type = 'LONG/LAT'
            zone = 0
            input_file_name = ||
            input_format = 1
            interpolation_type = 1
            use_end_time = 1
            first_time_step = 0
            last_time_step = 396
            time_step_frequency = 18
            number_of_points = 1
            [POINT_1]
               name = 'Point 1'
               x = 2.95
               y = 51.9363335
            EndSect  // POINT_1

            [LINE]
               npoints = 3
               x_first = -0.2
               y_first = 49.872667
               x_last = 6.1
               y_last = 54.0
            EndSect  // LINE

            [AREA]
               number_of_points = 4
               [POINT_1]
                  x = -1.682860599569629
                  y = 49.81812218993181
               EndSect  // POINT_1

               [POINT_2]
                  x = -1.682860599569629
                  y = 55.38173780835869
               EndSect  // POINT_2

               [POINT_3]
                  x = 8.956712006121856
                  y = 55.38173780835869
               EndSect  // POINT_3

               [POINT_4]
                  x = 8.956712006121856
                  y = 49.81812218993181
               EndSect  // POINT_4

               orientation = 0.0
               x_origo = -1.578551064219712
               x_ds = 0.5489975544732448
               x_npoints = 20
               y_origo = 49.87266744109285
               y_ds = 0.5489975544732448
               y_npoints = 11
               z_origo = -97.4107883362347
               z_ds = 15.56245389898787
               z_npoints = 10
            EndSect  // AREA

            [INTEGRAL_WAVE_PARAMETERS]
               Touched = 0
               type_of_spectrum = 1
               minimum_frequency = 0.033
               maximum_frequency = 1.061989502698701
               separation_of_wind_sea_and_swell = 3
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               hm0_minimum = 0.01
               type_of_h_max = 3
               duration = 10800.0
               distance_above_bed_for_particle_velocity = 0.0
               minimum_direction = 0.0
               maximum_direction = 360.0
               [Total_wave_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 1
                  Peak_wave_period = 1
                  Wave_period_t01 = 1
                  Wave_period_t02 = 1
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 1
                  Mean_wave_direction = 1
                  Directional_standard_deviation = 1
                  Wave_velocity_components = 1
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Total_wave_parameters

               [Wind_sea_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Wind_sea_parameters

               [Swell_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Swell_parameters

            EndSect  // INTEGRAL_WAVE_PARAMETERS

            [INPUT_PARAMETERS]
               Touched = 0
               Surface_elevation = 1
               Water_depth = 0
               Current_velocity_components = 1
               Wind_speed = 1
               Wind_direction = 1
               Ice_concentration = 0
            EndSect  // INPUT_PARAMETERS

            [MODEL_PARAMETERS]
               Touched = 0
               Wind_friction_speed = 0
               Roughness_length = 0
               Drag_coefficient = 0
               Charnock_constant = 0
               Friction_coefficient = 0
               Breaking_parameter_gamma = 0
               Courant_number = 0
               Time_step_factor = 0
               Convergence_angle = 0
               Length = 0
               Area = 0
               Threshold_period = 0
               Roller_area = 0
               Roller_dissipation = 0
               Breaking_index = 0
            EndSect  // MODEL_PARAMETERS

            [SPECTRAL_PARAMETERS]
               Touched = 0
               separation_of_wind_sea_and_swell = 3.0
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               wave_energy = 1
               wave_action = 0
               zeroth_moment_of_wave_action = 0
               first_moment_of_wave_action = 0
               wave_energy_wind_sea = 0
               wave_energy_swell = 0
            EndSect  // SPECTRAL_PARAMETERS

         EndSect  // OUTPUT_1

         [OUTPUT_2]
            Touched = 1
            include = 1
            title = 'points'
            file_name = 'ts_storm_4.dfs0'
            type = 1
            format = 0
            flood_and_dry = 2
            coordinate_type = 'LONG/LAT'
            zone = 0
            input_file_name = ||
            input_format = 1
            interpolation_type = 1
            use_end_time = 1
            first_time_step = 0
            last_time_step = 396
            time_step_frequency = 1
            number_of_points = 4
            [POINT_1]
               name = 'Europlatform'
               x = 3.276
               y = 51.999
            EndSect  // POINT_1

            [POINT_2]
               name = 'K14'
               x = 3.6333
               y = 53.2667
            EndSect  // POINT_2

            [POINT_3]
               name = 'F16'
               x = 4.0122
               y = 54.1167
            EndSect  // POINT_3

            [POINT_4]
               name = 'F3'
               x = 4.6939
               y = 54.8489
            EndSect  // POINT_4

            [LINE]
               npoints = 3
               x_first = -0.2
               y_first = 49.87266744109285
               x_last = 6.1
               y_last = 54.00000000000001
            EndSect  // LINE

            [AREA]
               number_of_points = 4
               [POINT_1]
                  x = -0.263
                  y = 49.83139411550378
               EndSect  // POINT_1

               [POINT_2]
                  x = -0.263
                  y = 54.04127332558908
               EndSect  // POINT_2

               [POINT_3]
                  x = 6.162999999999999
                  y = 54.04127332558908
               EndSect  // POINT_3

               [POINT_4]
                  x = 6.162999999999999
                  y = 49.83139411550378
               EndSect  // POINT_4

               orientation = 0.0
               x_origo = -1.578551064219712
               x_ds = 0.5489975544732448
               x_npoints = 20
               y_origo = 49.87266744109285
               y_ds = 0.5489975544732448
               y_npoints = 11
               z_origo = -97.4107883362347
               z_ds = 15.56245389898787
               z_npoints = 10
            EndSect  // AREA

            [INTEGRAL_WAVE_PARAMETERS]
               Touched = 0
               type_of_spectrum = 1
               minimum_frequency = 0.033
               maximum_frequency = 1.061989502698701
               separation_of_wind_sea_and_swell = 3
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               hm0_minimum = 0.01
               type_of_h_max = 3
               duration = 10800.0
               distance_above_bed_for_particle_velocity = 0.0
               minimum_direction = 0.0
               maximum_direction = 360.0
               [Total_wave_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Total_wave_parameters

               [Wind_sea_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Wind_sea_parameters

               [Swell_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Swell_parameters

            EndSect  // INTEGRAL_WAVE_PARAMETERS

            [INPUT_PARAMETERS]
               Touched = 1
               Surface_elevation = 0
               Water_depth = 0
               Current_velocity_components = 0
               Wind_speed = 1
               Wind_direction = 1
               Ice_concentration = 0
            EndSect  // INPUT_PARAMETERS

            [MODEL_PARAMETERS]
               Touched = 0
               Wind_friction_speed = 0
               Roughness_length = 0
               Drag_coefficient = 0
               Charnock_constant = 0
               Friction_coefficient = 0
               Breaking_parameter_gamma = 0
               Courant_number = 0
               Time_step_factor = 0
               Convergence_angle = 0
               Length = 0
               Area = 0
               Threshold_period = 0
               Roller_area = 0
               Roller_dissipation = 0
               Breaking_index = 0
            EndSect  // MODEL_PARAMETERS

            [SPECTRAL_PARAMETERS]
               Touched = 0
               separation_of_wind_sea_and_swell = 3.0
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               wave_energy = 1
               wave_action = 0
               zeroth_moment_of_wave_action = 0
               first_moment_of_wave_action = 0
               wave_energy_wind_sea = 0
               wave_energy_swell = 0
            EndSect  // SPECTRAL_PARAMETERS

         EndSect  // OUTPUT_2

      EndSect  // OUTPUTS

   EndSect  // SPECTRAL_WAVE_MODULE




   	[DATA_ASSIMILATION_MODULE]
      [TIME]
         start_time_step = 0

		 start_time_step_assimilation = 6 //(default=0) time to start assimilation
         time_step_factor_assimilation = 2 //refers to the overall time step
      EndSect  // TIME

	  [METHOD]
		type = 1               // 0=Free, 1=EnKF (ensemble), 2=Simple, 3=Steady, 4=EnOI 
		ensemble_size = 10
		algorithm = 1          // (for type=1) 1=serialESRF, 2=DEnKF, 3=ETKF
        Rfactor_relative = 3.0 // inflation only where model was updated (e.g. 2.0)		
		Rfactor_all = 1.0      // factor on st_dev for *all* measurements
	  EndSect  // FILTER
	  
      [MODEL_ERROR_MODEL]         
		use_clock_based_seeds = false // false=same seq. of random numbers every time
		number_of_model_errors = 1
		
		[MODEL_ERROR_1]
			type = 'wind'
				[Error_Formulation]	
				 st_dev = 1.5                        // [m] (if WL), [m/s] (flow/wind(uv)/wind speed)
				 
				 propagation_type = 'AR(1)'          // 'whitenoise', 'AR(1)'
				 propagation_parameter = 10800       // if AR(1): half-time in seconds
				 initialization_type = 1             // start: 0:from 0, 1:from st_dev
				 
				 horizontal_discretization_type = 2  // 0: constant, 1: (piecewise) linear, 2: equidistant grid
				 horizontal_grid_spacing = 80000     // if discr_type = 2 (in meters)

				 horizontal_corr_function = 1        // 1: gaussian, 2: exponential 
				 horizontal_corr = 500000			 // correlation length in meters
			EndSect
		EndSect  // MODEL_ERROR_1
		
      EndSect  // MODEL_ERROR_MODEL

      [MODEL_STATE_SPACE]  
		sw_state_space_type = 1	
		number_of_variable_transforms = 0
		[VARIABLE_TRANSFORM_1] 
			variable_id = 2      // Hm0 
			transform_type = 2   // 1=log, 2=square, -1=exp, -2=sqrt
		EndSect					 
      EndSect  // MODEL_STATE_SPACE

      [MEASUREMENTS]
         number_of_independent_measurements = 5

         [MEASUREMENT_1]
			include = 1             // 0: inactive, 1: active used for DA, 2: active but only for validation
            name = 'Hm0_EPL'
            measured_variable = 'SW%Hm0'  // 'water level' (default), 'u', 'v', 'ua', 'va' (3d), 'temperature'
            type = 1                      // 1=fixed station (dfs0); 2=track (dfs0)
            file_name = |.\obs\eur_Hm0_20171026-20171029_UTC.dfs0|
			
            item_number = 1 
			data_offset = 0.0             // add this amount to all data in file (default=0.0)
			type_time_interpolation = 1   // 0: discrete/no interp, 1: piecewise linear, 2: cubic spline, interpolates data between 2 assimilation time steps
			//time_window_in_seconds = 300  // in case type_time_interpolation=0 or type=2 (use AD time step size)
			
            position = 3.276,51.999
			coordinate_type = 'LONG/LAT'   // default: same as model
            group = 1                     // measurements may belong to group (used by localization)
            st_dev = 0.4
			
         EndSect  // MEASUREMENT_1
         [MEASUREMENT_2]
			include = 1             // 0: inactive, 1: active used for DA, 2: active but only for validation
            name = 'Hm0_K14'
            measured_variable = 'SW%Hm0'  // 'water level' (default), 'u', 'v', 'ua', 'va' (3d), 'temperature'
            type = 1                      // 1=fixed station (dfs0); 2=track (dfs0)
            file_name = |.\obs\k14_Hm0_20171026-20171029_UTC.dfs0|
			
            item_number = 1 
			data_offset = 0.0             // add this amount to all data in file (default=0.0)
			type_time_interpolation = 1   // 0: discrete/no interp, 1: piecewise linear, 2: cubic spline
			//time_window_in_seconds = 300  // in case type_time_interpolation=0 or type=2 (use AD time step size)
			
            position = 3.6333,53.2667 
			coordinate_type = 'LONG/LAT'   // default: same as model
            group = 1                     // measurements may belong to group (used by localization)
            st_dev = 0.4
			
         EndSect  // MEASUREMENT_2

         [MEASUREMENT_3]
		 	include = 2             // 0: inactive, 1: active used for DA, 2: active but only for validation
            name = 'Hm0_F16'
            measured_variable = 'SW%Hm0'  // 'water level' (default), 'u', 'v', 'ua', 'va' (3d), 'temperature'
            type = 1                      // 1=fixed station (dfs0); 2=track (dfs0)
            file_name = |.\obs\f16_Hm0_20171026-20171029_UTC.dfs0|
			
            item_number = 1 
			data_offset = 0.0             // add this amount to all data in file (default=0.0)
			type_time_interpolation = 0   // 0: discrete/no interp, 1: piecewise linear, 2: cubic spline
			//time_window_in_seconds = 300  // in case type_time_interpolation=0 or type=2 (use AD time step size)
			
            position = 4.0122,54.1167
			coordinate_type = 'LONG/LAT'   // default: same as model
            group = 1                     // measurements may belong to group (used by localization)
            st_dev = 0.7
			
         EndSect  // MEASUREMENT_3
		 
         [MEASUREMENT_4]
			include = 1             // 0: inactive, 1: active used for DA, 2: active but only for validation
            name = 'Hm0_F3'
            measured_variable = 'SW%Hm0'  // 'water level' (default), 'u', 'v', 'ua', 'va' (3d), 'temperature'
            type = 1                      // 1=fixed station (dfs0); 2=track (dfs0)
            file_name = |.\obs\f03_Hm0_20171026-20171029_UTC.dfs0|
			
            item_number = 1 
			type_time_interpolation = 1   // 0: discrete/no interp, 1: piecewise linear, 2: cubic spline

			//time_window_in_seconds = 300  // in case type_time_interpolation=0 or type=2 (use AD time step size)
			
            position = 4.6939,54.8489
			coordinate_type = 'LONG/LAT'   // default: same as model
            group = 1                     // measurements may belong to group (used by localization)
            st_dev = 0.4
			
         EndSect  // MEASUREMENT_4
		
         [MEASUREMENT_5]
			include = 2             // 0: inactive, 1: active used for DA, 2: active but only for validation
            name = 'Hm0_Cryosat2'
            measured_variable = 'SW%Hm0'  // 'water level' (default), 'u', 'v', 'ua', 'va' (3d), 'temperature'
            type = 2                      // 1=fixed station (dfs0); 2=track (dfs0)
            //file_name = |.\obs\RADS_North_Sea_rads_s3a_NS_2016-2019_polygon_all_params_20km-20171026-20171029.dfs0| //Sentinel3a
			file_name = |.\obs\RADS_North_Sea_rads_c2_Dutch_102017_all_params_20km-20171027-20171029.dfs0|   //Cryosat2			
            item_number = 4 
			data_offset = 0.0             // add this amount to all data in file (default=0.0)
			type_time_interpolation = 0   // 0: discrete/no interp, 1: piecewise linear, 2: cubic spline
			//time_window_in_seconds = 300  // in case type_time_interpolation=0 or type=2 (use AD time step size)
			
			coordinate_type = 'LONG/LAT'   // default: same as model
            group = 1                     // measurements may belong to group (used by localization)
            st_dev = 0.1
			
         EndSect  // MEASUREMENT_5

      EndSect  // MEASUREMENTS

      [ESTIMATOR]
         type = 1
         [REGULARIZATION]
            use_temporal_smoothing = true
            smoothing_halftime = 3600
            use_localization = false
            [LOCALIZATION]                 
               number_of_groups = 1
               [GROUP_1]
				  horizontal_corr_function = 3  // 1:gauss, 2:exp, 3:Gasperi&Cohn(default) 
				  horizontal_corr = 200000			// correlation length in meters			                       
               EndSect  // GROUP_1

            EndSect  // LOCALIZATION

         EndSect  // BLUE
      EndSect  // ESTIMATOR

	  [ERRCOV_IO]  // only for DA%algorithm=1
			
	       [INPUT]
               type = 0  // 0: not active
			EndSect  // INPUT
			
			[OUTPUTS]
			    number_of_outputs = 1
				
				[OUTPUT_1]				
					include = 1  
					//first_time_step = 100
					//last_time_step = 119520
					//time_step_frequency = 118520  // like time_step_factor
					time_average = true        // true=only a single step output
					smoothed_out = false       // true=output time-smoothed error_covariance (instead of normal error_cov)
					
					file_name_area = 'ErrCovIO_avg_Area.dfsu'					
					file_name_spectrum = 'ErrCovIO_avg_Spectrum.dfsu'
					file_name_err01_wind_u = 'ErrCovIO_avg_Wind_err_u.dfs2'
					file_name_err01_wind_v = 'ErrCovIO_avg_Wind_err_v.dfs2'				
				EndSect  // OUTPUT_1

			EndSect  // OUTPUTS
		
	  EndSect  // ERRCOV_IO

	  [ENSEMBLE_IO]
		
         [INPUT]  // can be used for hot-in for ensemble run
            type = 0  // 0: not active, 1: Meanstate, 2: Ensemble
            
			modelerror_only = false
            file_name_area = |.\EnsembleInput\State_Area_ens.dfsu| 
            file_name_spectrum = |.\EnsembleInput\State_Spectrum_ens.dfsu|
			//file_name_err01_wind_u = |.\EnsembleInput\State_Wind_err_u.dfs2|
            //file_name_err01_wind_v = |.\EnsembleInput\State_Wind_err_v.dfs2|	
			
         EndSect  // INPUT
		 
		 [OUTPUTS]
			number_of_outputs = 2
			[OUTPUT_1]  // can be used for hot-out for ensemble run
				type = 2  // 0: not active, 1: Meanstate, 2: Ensemble
				include_model_errors = true
				include_model_variables = true
				
				file_name_area     = 'State_Area_ens.dfsu'            
				file_name_spectrum = 'State_Spectrum_ens.dfsu'
				file_name_err01_wind_u = 'State_Wind_err_u.dfs2'
				file_name_err01_wind_v = 'State_Wind_err_v.dfs2'
			
				//first_time_step = 0
				//last_time_step = 6000
				time_step_frequency = 10
			EndSect // OUTPUT_1
						
			[OUTPUT_2]  // can be used for hot-out for ensemble run
				type = 1  // 0: not active, 1: Meanstate, 2: Ensemble
				include_model_errors = true
				include_model_variables = true
				
				is_stateupdate = true
				file_name_area     = 'StateUpdate_Area.dfsu'            
				file_name_spectrum = 'StateUpdate_Spectrum.dfsu'
				file_name_err01_wind_u = 'StateUpdate_Wind_err_u.dfs2'
				file_name_err01_wind_v = 'StateUpdate_Wind_err_v.dfs2'
			
				first_time_step = 18
				//last_time_step = 6000
				time_step_frequency = 6
			EndSect // OUTPUT_2

         EndSect  // OUTPUTS
	  EndSect  // ENSEMBLE_IO

	  [DIAGNOSTICS]
		[OUTPUTS]
			number_of_outputs = 7
			 
			 [OUTPUT_1]
			    include = 1
				type = 1 //type=1 refers to the measurements, type=2 any variables in state description 
				measurement_id = 1
				file_name = 'Diagnostics_EPL.dfs0'
			 EndSect  // OUTPUT_1
			 
			 [OUTPUT_2]
			    include = 1
				type = 1
				measurement_id = 2
				file_name = 'Diagnostics_K14.dfs0'
			 EndSect  // OUTPUT_2
			 
			 [OUTPUT_3]
			    include = 1
				type = 1
				measurement_id = 3
				file_name = 'Diagnostics_F16.dfs0'
			 EndSect  // OUTPUT_3

			 [OUTPUT_4]
			    include = 1
				type = 1
				measurement_id = 4
				file_name = 'Diagnostics_F3.dfs0'
			 EndSect  // OUTPUT_4

			 [OUTPUT_5] //example if other parameters have to be checked, e.g model errors at points
			    include = 0
				type = 2
				variable_name = 'err01_wind_u'
				position = 3, 48
				file_name = 'Diagnostics_wind_u_err.dfs0'
			 EndSect  // OUTPUT_5

			 [OUTPUT_6]
			    include = 1
				type = 1
				measurement_id = 5
				file_name = 'Diagnostics_C2.dfs0'
			 EndSect  // OUTPUT_6
			 
			 [OUTPUT_7] 
			    include = 1
				type = 3
				file_name = 'Global_stats.dfs0'
			 EndSect  // OUTPUT_7			 
			 
		EndSect  // OUTPUTS
		
	  EndSect  // DIAGNOSTICS	
	  
	EndSect  // DATA_ASSIMILATION_MODULE  

EndSect  // FemEngineSW

