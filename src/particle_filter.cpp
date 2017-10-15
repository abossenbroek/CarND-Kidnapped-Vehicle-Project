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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = NUM_PARTICLES;
  // Allocate memory for all the particles.
  particles.resize(NUM_PARTICLES);

  normal_distribution<double> noise_x(x, std[0]);
  normal_distribution<double> noise_y(y, std[1]);
  normal_distribution<double> noise_theta(theta, std[2]);

  // Work through all the particles to set the x, y, theta and noise.
  for (vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p) {
    p->x = noise_x(*rng_gen);
    p->y = noise_y(*rng_gen);
    p->theta = noise_theta(*rng_gen);
    p->weight = 1.;
    p->id = p - particles.begin();
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  // Work through all the particles to set the x, y, theta and noise.
  for (vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p) {
    if (fabs(yaw_rate) < 1.0e-2) {
      p->x += velocity * delta_t * cos(yaw_rate);
      p->y += velocity * delta_t * sin(yaw_rate);
    } else {
      p->x += velocity / yaw_rate * (sin(p->theta + yaw_rate * delta_t) - sin(p->theta));
      p->y += velocity / yaw_rate * (cos(p->theta) - cos(p->theta + yaw_rate * delta_t));
    }

    p->theta = yaw_rate * delta_t;

    p->x += noise_x(*rng_gen);
    p->y += noise_y(*rng_gen);
    p->theta += noise_theta(*rng_gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned obs_i = 0; obs_i < observations.size(); ++obs_i) {
    auto obs = observations[obs_i];

    double min_distance = INFINITY;
    int min_i = -1;
    for (unsigned i = 0; i < predicted.size(); ++i) {
      LandmarkObs pred_lm = predicted[i];
      double dx = (pred_lm.x - obs.x);
      double dy = (pred_lm.y - obs.y);
      double dist = dx*dx + dy*dy;
      if(dist < min_distance) {
        min_distance = dist;
        min_i = i;
      }
    }
    observations[obs_i].id = min_i;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double sigma_landmark[2] = {0.3, 0.3};

  for (vector<Particle>::iterator p_it = particles.begin(); p_it != particles.end(); ++p_it) {

    vector<LandmarkObs> predictions;

    for (auto l_it : map_landmarks.landmark_list) {
      double dx = l_it.x_f - p_it->x;
      double dy = l_it.y_f - p_it->y;

      // Determine whether the landmark is within the sensor observation range.
      if (dx * dx + dy * dy <= sensor_range * sensor_range) {
        // Add prediction
        predictions.push_back(LandmarkObs{l_it.id_i, l_it.x_f, l_it.y_f});
      }
    }

    vector<LandmarkObs> transformed_obs;
    for (auto o_it : observations) {
      // Transform from MAP to VEHICLE coordinate system
      double x_trans = p_it->x + cos(p_it->theta) * o_it.x - sin(p_it->theta) * o_it.y;
      double y_trans = p_it->y + sin(p_it->theta) * o_it.x + cos(p_it->theta) * o_it.y;
      // Store the value
      transformed_obs.push_back(LandmarkObs{o_it.id, x_trans, y_trans});
    }

    dataAssociation(predictions, transformed_obs);
    p_it->weight = 1.;

    for (unsigned i = 0; i < transformed_obs.size(); ++i) {
      auto obs = transformed_obs[i];
      auto pred = predictions[obs.id];

      double dist_x = pred.x - obs.x;
      double dist_y = pred.y - obs.y;
      double cov_x = sigma_landmark[0] * sigma_landmark[0];
      double cov_y = sigma_landmark[1] * sigma_landmark[1];

      // Calculate the weight for the give landmark
      double obs_w = (1. / (2. * M_PI * sigma_landmark[0] * sigma_landmark[1])) *
                     exp(-(dist_x * dist_x / (2. * cov_x) +
                           dist_y * dist_y / (2. * cov_y)));

      p_it->weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> new_particles;
  vector<double> weights;

  for (auto p_it : particles) {
    weights.push_back(p_it.weight);
  }

  // Generate random number for resampling algorithm
  uniform_int_distribution<int> uni_dist(0, num_particles - 1);
  int index = uni_dist(*rng_gen);

  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_real_distribution<double> uni_r_dist(0, max_weight);
  double beta = 0.;

  // Execute the resampling algorithm
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
