
#ifndef __gt_steinsampler_h__
#define __gt_steinsampler_h__

#include "image.h"
#include "transform.h"

#include "math/rng.h"

#include "dwi/tractography/GTStein/gt.h"
#include "dwi/tractography/GTStein/particle.h"
#include "dwi/tractography/GTStein/particlegrid.h"
#include "dwi/tractography/GTStein/energy.h"
#include "dwi/tractography/GTStein/spatiallock.h"
#include "../../mcmc/include/mcmc/samplers/stein_variational_gradient_descent_sampler.hpp"

namespace MR
{
  namespace DWI
  {
    namespace Tractography
    {
      namespace GTStein
      {

        /**
         * @brief The SteinSampler class
         */
        class SteinSampler
        {
          MEMALIGN(SteinSampler)
        public:
          SteinSampler(const Image<float> &dwi, Properties &p, Stats &s, ParticleGrid &pgrid,
                       EnergyComputer *e, Image<bool> &m)
              : props(p), stats(s), pGrid(pgrid), E(e), T(dwi),
                dims{size_t(dwi.size(0)), size_t(dwi.size(1)), size_t(dwi.size(2))},
                mask(m), lock(make_shared<SpatialLock<float>>(5 * Particle::L)),
                sigpos(Particle::L / 8.), sigdir(0.2)
          {
            //TODO
            // mcmc::stein_variational_gradient_descent_sampler<double, Eigen::VectorXf, Eigen::MatrixXf, std::normal_distribution<float>> sampler(
            //     [=](const Eigen::MatrixXf &state) {
            //       Eigen::MatrixXf gradients(state.rows(), state.cols());
            //       for (auto i = 0; i < state.rows(); ++i)
            //       {
            //         auto data = DWI::get_DW_scheme(Header(dwi));

            //         gradients(i, 0) = (data.array() - state(i, 0)).sum() / std::pow(state(i, 1), 2);
            //         gradients(i, 1) = (data.array() - state(i, 0)).pow(2).sum() / std::pow(state(i, 1), 3) - static_cast<float>(dwi.size()) / state(i, 1);
            //       }
            //       return gradients;
            //     },
            //     2,
            //     100,
            //     0.1);
            DEBUG("Initialise Stein Variational sampler.");
          }

          SteinSampler(const SteinSampler &other)
              : props(other.props), stats(other.stats), pGrid(other.pGrid), E(other.E->clone()),
                T(other.T), dims(other.dims), mask(other.mask), lock(other.lock), rng_uniform(), rng_normal(), sigpos(other.sigpos), sigdir(other.sigdir)
          {
            DEBUG("Copy Stein Variational sampler.");
          }

          ~SteinSampler() { delete E; }

          void execute();

          void next();

          void birth();
          void death();
          void randshift();
          void optshift();
          void connect();

        protected:
          Properties &props;
          Stats &stats;
          ParticleGrid &pGrid;
          EnergyComputer *E; // Polymorphic copy requires call to EnergyComputer::clone(), hence references or smart pointers won't do.

          Transform T;
          vector<size_t> dims;
          Image<bool> mask;

          std::shared_ptr<SpatialLock<float>> lock;
          Math::RNG::Uniform<float> rng_uniform;
          Math::RNG::Normal<float> rng_normal;
          float sigpos, sigdir;

          Point_t getRandPosInMask();

          bool inMask(const Point_t p);

          Point_t getRandDir();

          void moveRandom(const Particle *par, Point_t &pos, Point_t &dir);

          bool moveOptimal(const Particle *par, Point_t &pos, Point_t &dir) const;

          inline double calcShiftProb(const Particle *par, const Point_t &pos, const Point_t &dir) const
          {
            Point_t Dpos = par->getPosition() - pos;
            Point_t Ddir = par->getDirection() - dir;
            return gaussian_pdf(Dpos, sigpos) * gaussian_pdf(Ddir, sigdir);
          }

          inline double gaussian_pdf(const Point_t &x, double sigma) const
          {
            return std::exp(-x.squaredNorm() / (2 * sigma)) / std::sqrt(2 * Math::pi * sigma * sigma);
          }
        };

      } // namespace GTStein
    }   // namespace Tractography
  }     // namespace DWI
} // namespace MR

#endif // __gt_steinsampler_h__
