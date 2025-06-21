#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import requests
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import json
import datetime as dt
import mateomatics.api as api

class BallProperties:
	'''Golf ball physical properties'''
	mass: float = 0.0459		#kg
	diameter: float = 0.0427	#m
	drag_coefficient: float = 0.47	
	lift_coefficient: float = 0.25
	area: float
	
	def post_init(self):
		if self.area is None:
			self.area = np.pi * (self.diameter/2)**2
					
class LaunchParameters:
	''' Launch Conditions '''
	initial_speed: float	# m/s from club sensor
	launch_angle: float 	# deg horizontal
	azimuth_angle: float = 0.0	# deg vertical
	backspin_rpm: float = 2500.0	# RPM
	sidespin_rpm: float = 0.0
	
class EnvironmentalConditions:
	''' Environmental conditions from weather api'''
	temperature: float		# celsius
	pressure: float			# Pa
	humidity: float			# percent
	wind_speed: float		# m/s
	wind_direction: float	# deg (0=north, 90 = east)
	altitude: float			# m above sea level
	air_density: float
	
	def post_init(self):
		if self.air_density is None:
			self.air_density = self.calculate_air_density()
			
	def calculate_air_density(self) -> float:
		''' Calculate air density using ideal gas law'''
		R_dry = 287.058		# J/(kgK)
		R_vapor = 461.495	# ''''''
		
		# Convert Temperature to Kelvin
		T_kelvin = self.temperature + 273.15
		
		# Calculate saturation vapor pressure (Teten's formula)
		e_sat = 610.78*np.exp(17.27*self.temperature / (self.temperature + 237.3))
		
		# Calculate actual vapor pressure
		e_actual = (self.humidity / 100.0)*e_sat
		
		# partial pressure of dry air
		p_dry = self.pressure - e_actual
		
		# air density with humidity correction 
		rho = (p_dry/(R_dry*T_kelvin))+(e_actual/(R_vapor*T_kelvin))
		
		# altitude correction (simplified barometric form)
		if self.altitude > 0:
			rho *= np.exp(-self.altitude / 8400)
			
		return rho

class WeatherAPI:
	''' Interface for real-world weather data'''
	
	def get_weather_data(self, lat: float, lon: float, username = "bucknelluniversity_jorge_gherson", password = "o3J08RwN3u", altitude: float = 0.0):
		'''
		Fetch weather data from Meteomatics API.
		Returns simulation if no API key provided.
		'''
		if username and password:
			try:
				now = dt.utcnow().isoformat() + "Z"

				params = [
					"t_2m:C",
					"msl_pressure:hPa",
					"relative_humidity_2m:p",
					"wind_speed_10m:ms",
					"wind_dir_10m:d",
					"wind_gusts_10m_1h:ms",
					"precip_1h:mm",
				]

				base_url = "https://api.meteomatics.com"
				params_str = ','.join(params)
				coordinates = f"{lat},{lon}"
				url = f"{base_url}/{now}/{params_str}/{coordinates}/json"

				if altitude > 0:
					coordinates = f"{lat},{lon},{altitude}"
					url = f"{base_url}/{now}/{params_str}/{coordinates}/json"

				response = requests.get(url, auth=(username, password), timeout=10)
				response.raise_for_status()

				data = response.json()

				weather_data = {}

				for param_data in data["data"]:
					param_name = param_data["parameter"]
					value = param_data["coordinates"][0]["dates"][0]["value"]
					weather_data[param_name] = value

				wind_gust = weather_data.get('wind_gusts_10m_1h:ms', 0)

				return EnvironmentalConditions(
					temperature = weather_data.get["t_2m:C"],
					pressure = weather_data.get["msl_pressure:hPa"] * 100,  # convert hPa to Pa
					humidity = weather_data.get["relative_humidity_2m:p"],
					wind_speed = weather_data.get("wind_speed_10m:ms", 0),
					wind_direction = weather_data.get("wind_dir_10m:d", 0),
					altitude = weather_data.get(altitude) if altitude > 0 else 0,
					air_density = None  # will be calculated in EnvironmentalConditions
				)

			except requests.exceptions.RequestException as e:
				print(f"Meteomatics API network error: {e}")
				print("Using simulated weather data...")
			except KeyError as e:
				print(f"Meteomatics API data parsing error: {e}")
				print("Using simulated weather data...")
			except Exception as e:
				print(f"Meteomatics API error: {e}")
				print("Using simulated weather data...")

			return WeatherAPI.get_simulated_weater(self)
		
	def get_simulated_weater(self) -> EnvironmentalConditions:
		''' generate realistic weather conditions for sim'''
		return EnvironmentalConditions(
			temperature = 22.0 + np.random.uniform(-5, 8),	# 17 - 30 C
			pressure = 101345.0 + np.random.uniform(-2000.0, 2000.0), 
			humidity = 60.0 + np.random.uniform(-20, 30),
			wind_speed = np.random.uniform(0,8),
			wind_direction = np.random.uniform(0, 360)
		)
	
	def get_weather_with_uncertainty(self, lat: float, lon: float,  username = "bucknelluniversity_jorge_gherson", password = "o3J08RwN3u", altitude: float = 0.0) -> Dict:
		base_weather = WeatherAPI.get_weather_data(lat, lon, username, password, altitude)
		uncertainty = {
			"temperature_std": 0.5,
			"pressure_std": 50,
			"humidity_std": 3.0,
			"wind_speed_std": 0.3,
			"wind_direction_std": 5.0,
		}
		
		return {
			"base_conditions": base_weather, 
			"uncertainty": uncertainty
		}
class ClubSensor: 
	''' Simulated mounted sensor'''
	
	def measure_ball_speed(self)-> float:
		'''
		Simulate sensor measurement of ball initial speed. 
		In real implementation, this would interface with hardware
		'''
		base_speed = {
			"driver": 70.0,
			"7_iron": 55.0,
			"wedge": 46.0
		}
		
		base_speed = base_speed["driver"]
		measurement_noise = np.random.uniform(-2,2)
		
		return base_speed + measurement_noise

class GolfBallTrajectorySimulator:
	''' main simulation engine for golf ball flight simulation'''
	
	def init(self, ball: BallProperties, launch: LaunchParameters, env: EnvironmentalConditions):
		self.ball = ball
		self.launch = launch
		self.env = env
		
		# degrees to rad
		self.launch_angle_rad = np.radians(launch.launch_angle)
		self.azimuth_rad = np.radians(launch.azimuth_angle)
		self.wind_direction_rad = np.radians(env.wind_direction)
		
		# spin to angular velocity
		self.backspin_rads = launch.backspin_rpm * (2*np.pi / 60)
		self.sidespin_rads = launch.sidespin_rpm * (2*np.pi / 60)
		
		# initial velocity components
		self.v0_x = launch.initial_speed * np.cos(self.launch_angle_rad)*np.cos(self.azimuth_rad)
		self.v0_y = launch.initial_speed * np.cos(self.launch_angle_rad)*np.sin(self.azimuth_rad)
		self.v0_z = launch.initial_speed * np.sin(self.launch_angle_rad)
		
		# wind vel components
		self.wind_x = env.wind_speed * np.sin(self.wind_direction_rad)
		self.wind_y = env.wind_speed * np.cos(self.wind_direction_rad)
		self.wind_z = 0.0
		
	def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
		'''Differential equations for 3d golf ball motion with aerodynamics effects'''
		
		x, y, z, vx, vy, vz = state
		vx_rel = vx - self.wind_x
		vy_rel = vy - self.wind_y
		vz_rel = vz - self.wind_z
		
		v_rel_mag = np.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2)
		
		if v_rel_mag < 0.1: 	# avoid 0 div
			return np.array([vx, vy, vz, 0, 0, -9.81])
		
		# drag force
		drag = 0.5*self.env.air_density * self.ball.drag_coefficient*self.ball.area * v_rel_mag**3
		drag_x = -drag*(vx_rel / v_rel_mag) / self.ball.mass
		drag_y = -drag*(vy_rel / v_rel_mag) / self.ball.mass
		drag_z = -drag*(vz_rel / v_rel_mag) / self.ball.mass
		
		# magnus force (perp to vel and spin axis)
		magnus_force = 0.5*self.env.air_density * self.ball.lift_coefficient * self.ball.area * v_rel_mag**2
		
		# Backspin contribution
		lift_z = magnus_force * (self.backspin_rads / (self.backspin_rads +1))
		
		#sidespin contribution
		side_force = magnus_force * (self.backspin_rads) / (abs(self.sidespin_rads) +1)
		side_y = side_force* np.sin(self.sidespin_rads) / self.ball.mass
		
		# total accelerations
		ax = drag_x
		ay = drag_y + side_y
		az = drag_z + lift_z - 9.81		# account for gravity
		
		return np.array([vx, vy, vz, ax, ay, az])
		
	def simulate(self, tmax: float = 15.0) -> Dict:
		''' Execute the trajectory of the simulation'''
		init_state = np.array([0,0,0, self.v0_x, self.v0_y, self.v0_z])
		
		def hit_ground(t, state):
			return state[2]		#z coordinate
		hit_ground.terminal = True
		hit_ground.direction = -1
		
		sol = solve_ivp(
			self.equations_of_motion, 
			[0, tmax],
			init_state,
			events = hit_ground, 
			dense_output=True,
			rtol = 1e-8,
			atol= 1e-10
		)
		
		if not sol.success:
			raise RuntimeError("Simulation failed to converge")
		
		t_flight = sol.t_events[0][0] if sol.t_events[0].size > 0 else sol.t[-1]
		t_array = np.linspace(0, t_flight, 1000)
		trajectory = sol.sol(t_array)
		
		x_traj = trajectory[0]
		y_traj = trajectory[1]
		z_traj = trajectory[2]
		
		# key metrics
		max_height = max(z_traj)
		total_distance = np.sqrt(x_traj[-1]**2 + y_traj[-1]**2)
		landing_x = x_traj[-1]
		landing_y = y_traj[-1]
		
		return {
			"time": t_array,
			"x": x_traj,
			"y": y_traj,
			"z": z_traj,
			"flight_time": t_flight,
			"max_height": max_height,
			"total_distance": total_distance,
			"landing_point": (landing_x, landing_y),
			"trajectory_solution": sol
		}
		
	def calculate_landing_radius(self, num_simulations: int = 1000) -> Dict:
		'''
		Monte Carlo simulation to calculate landing radius
		'''
		landing_points = []
		
		orig_speed = self.launch.initial_speed
		orig_angle = self.launch.launch_angle
		orig_backspin = self.launch.backspin_rpm
		orig_wind_speed = self.env.wind_speed
		orig_wind_dir = self.env.wind_direction
		
		for i in range(num_simulations):
			# probable parameter variations
			speed_var  = np.random.normal(0, 0.5)
			angle_var = np.random.normal(0, 0.2)
			spin_var = np.random.normal(0, 50)
			wind_speed_var = np.random.normal(0, 0.3)
			wind_dir_var = np.random.normal(0, 5)
			
			# apply variations
			self.launch.initial_speed = orig_speed + speed_var
			self.launch.launch_angle = orig_angle + angle_var
			self.launch.backspin_rpm = orig_backspin + spin_var
			self.env.wind_speed = max(0, orig_wind_speed + wind_speed_var)
			self.env.wind_direction = (orig_wind_dir + wind_dir_var) % 360 		# periodic boundary 
			
			# update parameters 
			self.init(self.ball, self.launch, self.env)
			
			try:
				result = self.simulate()
				landing_points.append(result["landing_point"])
			except:
				continue	# skip failed sims
		
		# restore parameters
		self.launch.initial_speed = orig_speed
		self.launch.launch_angle = orig_angle 
		self.launch.backspin_rpm = orig_backspin
		self.env.wind_speed = orig_wind_speed
		self.env.wind_direction = orig_wind_dir
		
		# stats
		landing_points = np.array(landing_points)
		if len(landing_points) == 0:
			return {"error": "No successful simulations"}	# error prompt
		
		mean_x = np.mean(landing_points[:, 0])
		mean_y = np.mean(landing_points[:, 1])
		
		# distances from mean landing point
		distances = np.sqrt((landing_points[:,0]-mean_x)**2 + (landing_points[:,1]-mean_y)**2)
		
		# radius stats
		radius_50 = np.percentile(distances, 50)
		radius_90 = np.percentile(distances, 90)
		radius_95 = np.percentile(distances, 95)
		
		return {
			"landing_points": landing_points,
			"mean_landing": (mean_x, mean_y),
			"radius_50": radius_50,
			"radius_90": radius_90, 
			"radius_95": radius_95,
			"std_x": np.std(landing_points[:, 0]), 	# standard deviation x
			"std_y": np.std(landing_points[:, 1]), 	# standard deviation y
			"num_simulations": len(landing_points)
		}
	
	def run_complete_simulation(self, lat = 40.7128, lon: float = -74.0060, api_key: Optional[str] = None) -> Dict:
		'''
		Complete simulation pipeline w/ sensor input and weather API
		'''
		print("Golf Ball Trajectory Simulation Starting")
		print("=" * 50)
	
		# get sensor measurement
		print("Reading club sensor...")
		ball_speed = ClubSensor.measure_ball_speed(self)
		print(f"Initial ball speed: {ball_speed:.1f} m/s ({ball_speed*2.237:.1f} mph")
		
		# weather data
		print("\nFetching weather conditions...")
		weather = WeatherAPI.get_weather_data(lat, lon, username="bucknelluniversity_jorge_gherson", password="o3J08RwN3u")
		print(f"Temperature: {weather.temperature:.1f} C")
		print(f"Pressure: {weather.pressure/1000:.1f} hPa")
		print(f"Humidity: {weather.humidity:.1f}%")
		print(f"Wind: {weather.wind_speed:.1f} m/s at {weather.wind_direction:.0f}°")
		print(f"Air Density: {weather.air_density:.3f} kg/m^3")
		
		# set up simulation parameters
		ball = BallProperties()
		launch = LaunchParameters(
			initial_speed = ball_speed,
			launch_angle = 12.0, 		# typical driver angle
			backspin_rpm = 2500, 		# typical driver backspin
			azimuth_angle = 0.0 		# straight shot
		)
	
		# trajectory sim
		print("\nRunning trajectory simulation...")
		simulator = GolfBallTrajectorySimulator(ball, launch, weather)
		trajectory = simulator.simulate()
		
		# calculate landing radius
		print("Calculating landing radius distribution...")
		landing_stats = simulator.calculate_landing_radius(num_simulations = 500)
		
		# results 
		print("\n" + "=" *50)
		print("SIMULATION RESULTS")
		print("="*50)
		print(f"Flight time: {trajectory['flight_time']:.2f} seconds")
		print(f"Maximum height: {trajectory['max_height']:.1f} m")
		print(f"Total distance: {trajectory['total_distance']:.1f} m")
		print(f"Landing Point: ({trajectory['landing_point'][0]:.1f}, {trajectory['landing_point'][1]:.1f}) m")
		
		print(f"\nLANDING RADIUS ANALYSIS:")
		print(f"50% of shots within: {landing_stats['radius_50']:.1f} m")
		print(f"90% of shots within: {landing_stats['radius_90']:.1f} m")
		print(f"95% of shots within: {landing_stats['radius_95']:.1f} m")
		print(f"Standard Deviation X: ±{landing_stats['std_x']:.1f} m")
		print(f"Standard Deviation Y: ±{landing_stats['std_y']:.1f} m")
		
		return {
			"trajectory": trajectory,
			"landing_stats": landing_stats,
			"weather": weather, 
			"launch_params": launch, 
			"ball_properties": ball
		}
		

		
		
		
