#pragma once
// Even-even clover LogDet monomial with QUDA-fused FORCE ASSEMBLY.
//
// Phase 2 of the light-sector force-assembly port (v1 = ratio rungs + tail,
// Grid/util/QudaSchurOpForce.h).  The LogDet force is the ONE piece v1 left in
// Grid: d/dU of  S = -Nf ln det M_ee.  Grid's reference
// (QCDLogDetCompactCloverEOAction::deriv_gpu) computes, per unique mu<nu,
//   lambda = Tr_spin( sigma_munu . Mee^{-1} ),   dSdU = Nf.Sum(+/-2 csw).Cmunu(lambda)
// which is EXACTLY QUDA's fused sigma-trace chain:
//   computeCloverSigmaTrace (Tr_spin(sigma . A^{-1}) -> oprod)
//     -> cloverDerivative (the Cmunu clover-staple analog) -> updateMomentum.
// There is NO Wilson hop, NO sigma-oprod, NO fermion vectors and NO linear
// solve -- a strict subset of the v1 sigma-call (which passed
// sigma_trace_coeff=0 to switch this very term OFF).
//
// Coefficient (EMPIRICALLY PINNED 2026-07-12 at 16^3, CORRECTED 2026-07-13
// after V3 48^3 COMPARE exposed csw-power error — see deriv()):
//   sigma_trace_coeff = 0.125 . Nf . csw . kappa,   UNIT unpack.
// NOTE: NOT the QUDA-canonical seed 2.Nf.kappa.csw with the v1 -1/(8 kappa^2)
// unpack — that seed gave cos=-1 and factor=2.5/(csw.kappa^2): the -1/(8k^2)
// factor belongs to the v1 SPINOR-pack pieces and is spurious for the pure
// trace (no spinor packing here).  QUDA's A^{-1} = 2 kappa Mee^{-1}, so QUDA
// and Grid forces are PROPORTIONAL (cos=1 verified exactly); the scalar was
// pinned to factor=0.99999996 at Nf=1,2 at 16^3 (csw=1.2493).
// The original csw^2 formula gave factor=0.9648 at 48^3 (csw=1.2054): the
// 0.9648 = csw_48/csw_16 ratio proved the formula was off by one power of csw.
// Corrected formula csw^1 with C=0.125 = 0.10005525*csw_16 gives factor~1.0000
// at both csw values (residual <3e-5 at 48^3).
//
// Parity (2026-10-02, fourth constructor argument, default Even = everything above, unchanged):
// Odd carries -Nf ln det M_oo instead, the partner of an EVEN-checkerboard pseudofermion (the
// QUDA_FORCE strange, matpc EVEN_EVEN_ASYMMETRIC).  The base class evaluates S() and the Grid
// force on M_oo; the QUDA call traces on the odd block via matpc EVEN_EVEN_ASYMMETRIC; the
// coefficient is the same (__docs/2026_10_02_strange_logdet_parity_mismatch.md).
//
// Residency: this monomial owns its clover.  The base LogDet action does not
// load QUDA; and a neighbouring rung solver's resident clover is at a DIFFERENT
// mass (kappa enters the clover), so we cannot borrow it.  We build our own
// param-source/loader (QudaCloverMultiShiftInverter at THIS operator's mass)
// and call loader_.SetGauge(U) at deriv entry -- loadGaugeQuda + loadCloverQuda
// (inverse included; refreshes ::extendedGaugeResident) at the correct kappa.
//
// Env knobs (default off => bit-identical to the Grid base action):
//   HASEN_QUDA_FORCE_LOGDET (driver-side gate; selects this subclass)
//   QUDA_LOGDET_FORCE_OFF=1        -> fall through to Grid base deriv (A/B)
//   QUDA_FORCE_KERNEL_COMPARE=1    -> also run Grid deriv, print cos/factor,
//                                     return the Grid force (physics-safe)
//   QUDA_LOGDET_FORCE_SIGMA_SIGN=<d> multiplies the unpacked force (default +1)
//   QUDA_LOGDET_FORCE_COEFF=<d>      overrides the coefficient scale (default 1)
//   QUDA_RUNG_FORCE_VERBOSE=1        -> per-call phase timers

#ifdef GRID_HAVE_QUDA

#include <Grid/qcd/action/pseudofermion/QCDLogDetCompactCloverEOAction.h>
#include <Grid/algorithms/iterative/QudaCloverInverter.h>          // QudaCloverParams
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaInit.h>
#include <Grid/util/QudaForcePrimitives.h>
#include <Grid/util/QudaPackGpu.h>
#include <Grid/util/QudaSchurOpForce.h>                            // CompareForcePiece

#include <cmath>
#include <cstdlib>
#include <memory>

NAMESPACE_BEGIN(Grid);

template <class Impl, class CloverHelpers = Grid::CompactCloverHelpers<Impl>>
class QCDLogDetCompactCloverEOQudaForceAction
    : public QCDLogDetCompactCloverEOAction<Impl, CloverHelpers> {
 public:
  typedef QCDLogDetCompactCloverEOAction<Impl, CloverHelpers> Base;
  typedef typename Base::FermionOperator FermionOperator;
  typedef typename Impl::GaugeField GaugeField;

  // parity: the clover block whose log-det this monomial carries, forwarded to the Grid base
  // (S(), the compare/fallback deriv).  Even (default) = the historical even block, byte-identical;
  // Odd = M_oo, the partner of an EVEN-checkerboard pseudofermion (QUDA_FORCE strange,
  // det S_ee = det M / det M_oo; __docs/2026_10_02_strange_logdet_parity_mismatch.md).
  QCDLogDetCompactCloverEOQudaForceAction(FermionOperator &Op, int nf,
                                          const QudaCloverParams &qp, int parity = Even)
      : Base(Op, nf, parity), nf_(nf) {
    Quda::initialize(-1, nullptr, Op.GaugeGrid());
    QudaCloverMultiShiftSpec spec;
    spec.shifts     = {0.0};
    spec.tols       = {1e-8};
    // computeCloverLogDetForceQuda (Grid/util/QudaForcePrimitives.h) traces Tr(σ·A^{-1}) on the
    // parity OPPOSITE to the matpc and accepts both asymmetric types: ODD_ODD -> EVEN block
    // (historical), EVEN_EVEN -> ODD block.  The matpc only feeds that parity choice here (this
    // loader never solves); loadCloverQuda builds the clover on both parities either way (the
    // inverse stored, or under QUDA_CLOVER_DYNAMIC formed inside the trace kernel at the traced
    // parity), and sigma_trace_coeff in deriv() does not depend on the parity.
    spec.matpc_type = (parity == Odd) ? QUDA_MATPC_EVEN_EVEN_ASYMMETRIC   // -> trace on ODD block
                                      : QUDA_MATPC_ODD_ODD_ASYMMETRIC;   // -> trace on EVEN block
    QudaCloverParams p = qp;
    p.use_multigrid = false;
    loader_ = std::make_unique<QudaCloverMultiShiftInverter>(
        Op.GaugeGrid(), p, spec);
  }

  std::string action_name() override {
    return Base::action_name() + " [QUDA force assembly]";
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    if (std::getenv("QUDA_LOGDET_FORCE_OFF") != nullptr) {
      Base::deriv(U, dSdU);  // Grid assembly (same-binary A/B)
      return;
    }
    const bool compare = std::getenv("QUDA_FORCE_KERNEL_COMPARE") != nullptr;

    // Own the residency: gauge + clover (with inverse) at THIS operator's mass.
    loader_->SetGauge(U);

    QudaInvertParam &inv_param = loader_->InvertParam();
    const double kappa = inv_param.kappa;
    const double csw   = inv_param.clover_csw;

    auto env_double = [](const char *name, double dflt) {
      if (const char *s = std::getenv(name); s && *s) return std::atof(s);
      return dflt;
    };
    const double coeff_scale = env_double("QUDA_LOGDET_FORCE_COEFF", 1.0);
    const double sigma_sign  = env_double("QUDA_LOGDET_FORCE_SIGMA_SIGN", 1.0);
    // Coefficient EMPIRICALLY PINNED 2026-07-12 at 16^3, CORRECTED 2026-07-13.
    // V3 48^3 COMPARE gave factor=0.9648 = csw_48/csw_16, proving the original
    // csw^2 formula was off by one power.  Corrected to csw^1: C=0.125 =
    // 0.10005525*csw_16 is exact at 16^3 and gives factor~1.00000 at 48^3.
    // NOTE: -1/(8 kappa^2) unpack is NOT applied here (no spinor packing in the
    // pure sigma-trace; that factor belongs to the v1 Wilson/sigma-oprod pieces).
    const double sigma_trace_coeff =
        0.125 * double(nf_) * csw * kappa * coeff_scale;
    const double quda_to_grid_factor = 1.0;  // trace term: no -1/(8 kappa^2)

    // MILC RECONSTRUCT_10 momentum scratch + EO permutation table.
    GridBase *full_grid = dSdU.Grid();
    const uint64_t V = Quda::local_volume(full_grid);
    constexpr int MOM_RECON = 10;
    if (scratch_.mom_dev.size() < V * 4 * MOM_RECON)
      scratch_.mom_dev.resize(V * 4 * MOM_RECON);
    if (!scratch_.eo_built) {
      Quda::BuildEoTable(full_grid, full_grid->LocalDimensions(),
                         scratch_.eo_table_dev);
      scratch_.eo_built = true;
    }

    QudaGaugeParam force_gauge_param = loader_->GaugeParam();
    force_gauge_param.type              = QUDA_GENERAL_LINKS;
    force_gauge_param.reconstruct       = QUDA_RECONSTRUCT_NO;
    force_gauge_param.gauge_order       = QUDA_MILC_GAUGE_ORDER;
    force_gauge_param.location          = QUDA_CUDA_FIELD_LOCATION;
    force_gauge_param.overwrite_mom     = 1;
    force_gauge_param.use_resident_mom  = 0;
    force_gauge_param.make_resident_mom = 0;
    force_gauge_param.return_result_mom = 1;

    QudaTwistFlavorType saved_twist = inv_param.twist_flavor;
    inv_param.twist_flavor = QUDA_TWIST_NO;

    auto t0 = usecond();
    Quda::computeCloverLogDetForceQuda(&scratch_.mom_dev[0], sigma_trace_coeff,
                                       &force_gauge_param, &inv_param);
    accelerator_barrier();  // QUDA mom visible before Grid unpack reads
    t_call_us_ += usecond() - t0;

    inv_param.twist_flavor = saved_twist;

    auto t1 = usecond();
    Quda::GpuUnpackMomToGauge(&scratch_.mom_dev[0], &scratch_.eo_table_dev[0], dSdU,
                             sigma_sign * quda_to_grid_factor);
    t_unpack_us_ += usecond() - t1;
    n_calls_++;

    if (std::getenv("QUDA_RUNG_FORCE_VERBOSE") != nullptr) PrintTimers();

    if (!compare) return;

    // COMPARE: Grid LogDet force from the base action, Ta-projected inside
    // CompareForcePiece (QUDA's updateMomentum output is traceless-anti-Herm).
    GaugeField A(dSdU.Grid());
    Base::deriv(U, A);
    Quda::CompareForcePiece(action_name() + " logdet", A, dSdU);
    dSdU = A;  // physics-safe: return the Grid force while comparing
  }

  ~QCDLogDetCompactCloverEOQudaForceAction() { PrintTimers(); }

 private:
  void PrintTimers() {
    if (n_calls_ == 0) return;
    std::cout << GridLogMessage << "[QudaLogDetForce/" << Base::action_name()
              << "] " << n_calls_ << " calls (ms/call): trace+deriv="
              << double(t_call_us_) * 1e-3 / n_calls_
              << " unpack=" << double(t_unpack_us_) * 1e-3 / n_calls_
              << std::endl;
  }

  int nf_;
  std::unique_ptr<QudaCloverMultiShiftInverter> loader_;
  Quda::QudaSchurForceScratch scratch_;
  uint64_t n_calls_ = 0, t_call_us_ = 0, t_unpack_us_ = 0;
};

NAMESPACE_END(Grid);

#endif  // GRID_HAVE_QUDA
