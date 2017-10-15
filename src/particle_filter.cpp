/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Add random Gaussian noise to each particle.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  weights.resize(NUM_PARTICLES, 1.0f);
  num_particles = NUM_PARTICLES;

  for (unsigned i = 0; i < NUM_PARTICLES; i++) {
    Particle p;
    p.x = dist_x(*rng_gen);
    p.y = dist_y(*rng_gen);
    p.theta = dist_theta(*rng_gen);
    p.id = i;
    p.weight = 1.;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  // Work through all the particles to set the x, y, theta and noise.
  for (vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p) {
    if (fabs(yaw_rate) < 1.0e-5) {
      p->x += velocity * delta_t * cos(p->theta);
      p->y += velocity * delta_t * sin(p->theta);
    } else {
      p->x += velocity / yaw_rate * (sin(p->theta + yaw_rate * delta_t) - sin(p->theta));
      p->y += velocity / yaw_rate * (cos(p->theta) - cos(p->theta + yaw_rate * delta_t));
      p->theta += yaw_rate * delta_t;
    }

    p->x += noise_x(*rng_gen);
    p->y += noise_y(*rng_gen);
    p->theta += noise_theta(*rng_gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (unsigned obs_i = 0; obs_i < observations.size(); ++obs_i) {
    LandmarkObs obs = observations[obs_i];

    double min_distance = INFINITY;
    int min_i = -1;
    for (unsigned i = 0; i < predicted.size(); ++i) {
      double dx = (predicted[i].x - obs.x);
      double dy = (predicted[i].y - obs.y);
      double dist = dx*dx + dy*dy;
      if(dist < min_distance) {
        min_distance = dist;
        min_i = i;
      }
    }
    observations[obs_i].id = min_i;
  }
}

const LandmarkObs vehicle_to_map(const LandmarkObs &obs, const Particle &p) {
  LandmarkObs out;

  // First rotate the local coordinates to the right orientation
  out.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
  out.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
  out.id = obs.id;
  return out;
}

inline const double gaussian_2d(const LandmarkObs &obs, const LandmarkObs &lm, const double sigma[]) {
  double cov_x = sigma[0] * sigma[0];
  double cov_y = sigma[1] * sigma[1];
  double normalizer = 2.0 * M_PI * sigma[0] * sigma[1];
  double dx = (obs.x - lm.x);
  double dy = (obs.y - lm.y);

  return exp(-(dx * dx / (2 * cov_x) + dy * dy / (2 * cov_y))) / normalizer;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  for (unsigned p_ctr = 0; p_ctr < particles.size(); ++p_ctr) {
    std::vector<LandmarkObs> predicted_landmarks;

    for (auto lm : map_landmarks.landmark_list) {
      double dx = lm.x_f - particles[p_ctr].x;
      double dy = lm.y_f - particles[p_ctr].y;

      // Add only if in range
      if (dx * dx + dy * dy <= sensor_range * sensor_range)
        predicted_landmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
    }
    std::vector<LandmarkObs> transformed_obs;
    double total_prob = 1.0f;

    // transform coordinates of all observations (for current particle)
    for (auto obs_lm : observations) {
      auto obs_map = vehicle_to_map(obs_lm, particles[p_ctr]);
      transformed_obs.push_back(std::move(obs_map));
    }
    // Stores index of associated landmark in the observation
    dataAssociation(predicted_landmarks, transformed_obs);

    for (unsigned i = 0; i < transformed_obs.size(); ++i) {
      LandmarkObs obs = transformed_obs[i];
      LandmarkObs assoc_lm = predicted_landmarks[obs.id];

      double pdf = gaussian_2d(obs, assoc_lm, std_landmark);
      total_prob *= pdf;
    }
    particles[p_ctr].weight = total_prob;
    weights[p_ctr] = total_prob;
  }

}

void ParticleFilter::resample() {
  std::vector<Particle> new_particles;

  // Generate random number for resampling algorithm
  uniform_int_distribution<int> uni_dist(0, num_particles - 1);
  int index = uni_dist(*rng_gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> uni_r_dist(0, max_weight);
  double beta = 0.;

  // Execute the resampling algorithm as described here
  // http://ais.informatik.uni-freiburg.de/teaching/ss14/robotics/slides/12-pf-mcl.pdf
  for (int i = 0; i < num_particles; ++i) {
    beta += uni_r_dist(*rng_gen) * 2.;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  // Copy the new particles to the particles
  particles = move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
