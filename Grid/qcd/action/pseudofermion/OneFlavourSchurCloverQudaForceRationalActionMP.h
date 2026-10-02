#pragma once
// QUDA-accelerated rational action with QUDA-computed gauge force.
//
// This is the "Phase 6" target: replace not just the inner multishift
// CG (Phase 5) but the entire per-pole gauge-derivative chain with one
// QUDA call to computeCloverForceQuda — which batches the Wilson hop
// derivative AND the σ_μν·F_μν clover derivative across all rational
// poles in a single fused GPU kernel chain.
//
// Built on top of OneFlavourSchurCloverRationalActionEven because
// computeCloverForceQuda hardcodes EVEN_EVEN_ASYMMETRIC matpc and
// expects EVEN-parity X_k inputs.
//
// Refresh and S inherit from the Grid-side base — Phase 5 already
// validated those paths against the FD test.  Only deriv() is overridden.

#include <Grid/qcd/action/pseudofermion/OneFlavourSchurCloverRationalActionEven.h>
#include <Grid/algorithms/iterative/QudaCloverMultiShiftInverter.h>
#include <Grid/util/QudaFieldConvert.h>

#include <quda.h>

namespace Grid {

template <class ImplD, class ImplF,
          class FermOpD_ = WilsonCloverFermion<ImplD, CloverHelpers<ImplD>>,
          class FermOpF_ = WilsonCloverFermion<ImplF, CloverHelpers<ImplF>>>
class OneFlavourSchurCloverQudaForceRationalActionMP
    : public OneFlavourSchurCloverRationalActionEven<ImplD, FermOpD_> {
 public:
  typedef OneFlavourSchurCloverRationalActionEven<ImplD, FermOpD_> Base;
  typedef typename Base::FermionField FermionField;
  typedef FermOpD_ FermOpD;
  typedef FermOpF_ FermOpF;
  typedef typename ImplD::GaugeField GaugeField;

  OneFlavourSchurCloverQudaForceRationalActionMP(
      FermOpD &opD, FermOpF & /*opF*/, GridBase * /*sp_rbgrid*/,
      OneFlavourRationalParams &p,
      const QudaCloverParams &qp,
      int /*reliable_update_freq*/ = 50)
    : Base(opD, p), qp_(qp) {

    QudaCloverMultiShiftSpec spec;
    spec.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    auto &poles = this->PowerNegHalf.poles;
    spec.shifts.resize(poles.size());
    spec.tols.assign(poles.size(), p.tolerance);
    for (size_t k = 0; k < poles.size(); ++k) spec.shifts[k] = poles[k];

    // Pass the gauge grid so QUDA inherits Grid's MPI communicator + rank map
    // (MPI-comms build).  Without a comm, QUDA uses a default rank order that
    // disagrees with Grid for >=2 partitioned directions -> wrong-neighbor halos
    // -> few-% force error.  GridBase is-a CartesianCommunicator.
    Quda::initialize(/*device=*/-1, /*mpi_dims=*/nullptr, opD.GaugeGrid());
    quda_ms_.reset(new QudaCloverMultiShiftInverter(
        opD.GaugeGrid(), qp_, spec));
    std::cout << GridLogMessage
              << "[OneFlavourSchurCloverQudaForceRationalActionMP] "
              << "built with " << poles.size()
              << " rational shifts, matpc=EVEN_EVEN_ASYMMETRIC" << std::endl;
  }

  // Override: solve via QUDA multishift.  When QUDA_FORCE_KERNEL=1 (env)
  // also use computeCloverForceQuda for the gauge-deriv chain; otherwise
  // (default during debugging) fall back to Grid's per-pole deriv chain
  // for the gauge force, which validates the EVEN-parity multishift
  // independently of the QUDA force routine.
  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    auto &FermOp = this->FermOp;
    auto &PhiEven = this->PhiEven;
    auto &PowerNegHalf = this->PowerNegHalf;
    const int Npole = PowerNegHalf.poles.size();
    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    GridBase *ggrid   = FermOp.GaugeGrid();

    FermOp.ImportGauge(U);
    quda_ms_->SetGauge(U);

    std::vector<FermionField> MPhi_k(Npole, fcbgrid);
    bool quda_force_kernel = (std::getenv("QUDA_FORCE_KERNEL") != nullptr);
    // For Path B (QUDA force kernel): keep solutions resident so
    // computeCloverForceQuda consumes them directly via
    // use_resident_solution=1 — bypasses host-side X pack/unpack.
    // make_resident_solution=1 in QUDA skips host download of multishift
    // solutions — MPhi_k host buffers come back as zero.  We need them to
    // be written either way (a) so Path A (Grid deriv) sees them and (b)
    // so we can pack a host X buffer for computeCloverForceQuda below.
    quda_ms_->solve_rb_even(PhiEven, MPhi_k, /*make_resident=*/false);

    // ------------------------------------------------------------------
    // Path A: Grid deriv chain.  Always run when QUDA_FORCE_KERNEL_COMPARE
    // is set, so we can compare to Path B's force in the same call and
    // empirically find the QUDA→Grid force normalization.
    // ------------------------------------------------------------------
    bool path_a_compare = std::getenv("QUDA_FORCE_KERNEL_COMPARE") != nullptr;
    if (std::getenv("QUDA_FORCE_KERNEL") == nullptr || path_a_compare) {
      SchurDifferentiableOperator<ImplD> Mpc(FermOp);
      FermionField X(fcbgrid), Y(fcbgrid);
      GaugeField tmp(ggrid);
      dSdU = Zero();
      for (int k = 0; k < Npole; ++k) {
        RealD ak = PowerNegHalf.residues[k];
        X = MPhi_k[k];
        Mpc.Mpc(X, Y);

        Mpc.MpcDeriv(tmp, Y, X);     dSdU = dSdU + ak * tmp;
        Mpc.MpcDagDeriv(tmp, X, Y);  dSdU = dSdU + ak * tmp;

        FermOp.MeeDeriv(tmp, Y, X, DaggerNo);    dSdU = dSdU + ak * tmp;
        FermOp.MeeDeriv(tmp, X, Y, DaggerYes);   dSdU = dSdU + ak * tmp;

        FermionField W_o(fcbgrid), Z_o(fcbgrid), tmp1(fcbgrid);
        FermOp.Meooe(X, tmp1);          FermOp.MooeeInv(tmp1, W_o);
        FermOp.MeooeDag(Y, tmp1);       FermOp.MooeeInvDag(tmp1, Z_o);
        FermOp.MooDeriv(tmp, Z_o, W_o, DaggerNo);   dSdU = dSdU + ak * tmp;
        FermOp.MooDeriv(tmp, W_o, Z_o, DaggerYes);  dSdU = dSdU + ak * tmp;
      }
      if (!path_a_compare) return;  // Path-A-only mode, done.
      std::cout << GridLogMessage
                << "[QudaForce] PathA dSdU norm2=" << norm2(dSdU) << std::endl;
    }
    GaugeField dSdU_pathA(ggrid);
    if (path_a_compare) dSdU_pathA = dSdU;
    // ------------------------------------------------------------------
    // Path B: QUDA's computeCloverForceQuda — fused force routine.
    // ------------------------------------------------------------------

    // Pack each MPhi_k (Even RB grid) into a flat half-volume host buffer
    // for QUDA.  Grid's RB cb-site order matches QUDA's cb_site within a
    // parity, so unvectorize → memcpy is a direct contiguous copy.
    //
    // KAPPA RESCALE: solve_rb_even returns X_grid = (M_pc_grid†·M_pc_grid +
    // σ_grid)⁻¹·b in mass form (M_pc_grid mass-form Schur).  But
    // computeCloverForceQuda interprets X as QUDA's kappa-form solution
    // X_kappa = (M_pc_kappa†·M_pc_kappa + σ_kappa)⁻¹·b.  Since
    // M_pc_kappa = 2κ·M_pc_grid → X_kappa = X_grid / (4κ²).  Scale here
    // so QUDA gets the convention it expects.
    int V_eo = Quda::local_volume(ggrid) / 2;
    using SiteSpinor = typename FermionField::scalar_object;
    static_assert(sizeof(SiteSpinor) == 24 * sizeof(double),
                  "expected 24 doubles/site for fermion");
    std::vector<std::vector<double>> x_bufs(Npole, std::vector<double>(24 * V_eo));
    std::vector<void *> x_ptrs(Npole);
    // X_grid (what solve_rb_even returns) equals X_kappa_form numerically:
    // QUDA's internal kappa-form solve, after the 4κ² rescale on src and
    // offsets, returns the SAME vector as Grid's mass-form X.  So no extra
    // rescale needed when feeding to computeCloverForceQuda.
    for (int k = 0; k < Npole; ++k) {
      std::vector<SiteSpinor> scalars;
      unvectorizeToLexOrdArray(scalars, MPhi_k[k]);
      std::memcpy(x_bufs[k].data(), scalars.data(),
                  V_eo * 24 * sizeof(double));
      x_ptrs[k] = x_bufs[k].data();
    }
    // X=0 diagnostic: if QUDA_FORCE_DBG_XZERO=1, force all X buffers to zero.
    if (std::getenv("QUDA_FORCE_DBG_XZERO")) {
      for (int k = 0; k < Npole; ++k) std::fill(x_bufs[k].begin(), x_bufs[k].end(), 0.0);
    }

    // Force / momentum buffer: QUDA writes ASQTAD_MOM_LINKS, reconstruct=10
    // (anti-Hermitian traceless 3×3 packed in 10 reals per site/dir).
    // With gauge_order = QUDA_MILC_GAUGE_ORDER, the buffer is a single
    // contiguous block, layout [site_eo][dir][10 reals].
    int V = Quda::local_volume(ggrid);
    constexpr int MOM_RECON = 10;
    std::vector<double> mom_buf(V * 4 * MOM_RECON, 0.0);

    // QUDA's inv_param comes from the multishift inverter (same κ/csw,
    // EVEN_EVEN_ASYMMETRIC matpc).  But the gauge_param for
    // computeCloverForceQuda is the param for the *momentum output*, not
    // the resident gauge — it must be QUDA_GENERAL_LINKS (so QUDA's
    // gauge_field machinery sets up extended ghosts correctly).  MILC
    // uses a fresh QudaGaugeParam (`newMILCGaugeParam(...,
    // QUDA_GENERAL_LINKS)`) for this call.
    QudaInvertParam &inv_param  = quda_ms_->InvertParam();
    int saved_use_resident             = inv_param.use_resident_solution;
    QudaDagType saved_dagger           = inv_param.dagger;
    QudaTwistFlavorType saved_twist_fl = inv_param.twist_flavor;
    double saved_mu                    = inv_param.mu;
    double saved_epsilon               = inv_param.epsilon;
    QudaPreserveSource saved_preserve  = inv_param.preserve_source;
    inv_param.use_resident_solution = 0;
    // PyQUDA's CloverWilsonAction.force() sets dagger=YES before
    // computeCloverForceQuda; this is the canonical Schur EE-asymmetric
    // convention. Validated on hot 4⁴ (2026-05-03): cos(Ta(A),B)=1.0
    // element-by-element vs Path A's Ta projection (was 0.894 with
    // dagger=NO). QUDA_FORCE_DAGGER_NO=1 reverts to the legacy NO setting
    // for regression debugging.
    inv_param.dagger = std::getenv("QUDA_FORCE_DAGGER_NO")
                           ? QUDA_DAG_NO
                           : QUDA_DAG_YES;
    inv_param.twist_flavor          = QUDA_TWIST_NO;
    inv_param.mu                    = 0.0;
    inv_param.epsilon               = 0.0;
    inv_param.preserve_source       = QUDA_PRESERVE_SOURCE_NO;
    inv_param.tm_rho                = 0.0;
    inv_param.clover_rho            = 0.0;
    inv_param.distance_pc_alpha0    = 0.0;
    inv_param.distance_pc_t0        = -1;
    inv_param.Ls                    = 1;
    QudaPrecision saved_sl  = inv_param.cuda_prec_sloppy;
    QudaPrecision saved_rs  = inv_param.cuda_prec_refinement_sloppy;
    QudaPrecision saved_pc  = inv_param.cuda_prec_precondition;
    inv_param.cuda_prec_sloppy            = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param.cuda_prec_precondition      = QUDA_DOUBLE_PRECISION;
    QudaGammaBasis saved_basis = inv_param.gamma_basis;
    if (std::getenv("QUDA_FORCE_DBG_UKQCD"))
      inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
    else if (std::getenv("QUDA_FORCE_DBG_CHIRAL"))
      inv_param.gamma_basis = QUDA_CHIRAL_GAMMA_BASIS;
    inv_param.clover_cuda_prec_sloppy            = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
    inv_param.clover_cuda_prec_precondition      = QUDA_DOUBLE_PRECISION;
    if (std::getenv("QUDA_FORCE_DBG_VERBOSE")) {
      inv_param.verbosity = QUDA_VERBOSE;
    }
    std::cout << GridLogMessage
              << "[QudaForce] inv_param state: kappa=" << inv_param.kappa
              << " csw=" << inv_param.clover_csw
              << " clover_coeff=" << inv_param.clover_coeff
              << " mass=" << inv_param.mass
              << " mu=" << inv_param.mu
              << " epsilon=" << inv_param.epsilon
              << " dslash_type=" << (int)inv_param.dslash_type
              << " matpc=" << (int)inv_param.matpc_type
              << " gamma_basis=" << (int)inv_param.gamma_basis
              << std::endl;

    QudaGaugeParam force_gauge_param = quda_ms_->GaugeParam();
    force_gauge_param.type        = QUDA_GENERAL_LINKS;
    force_gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    // MILC gauge_order — matches what MILC's qudaCloverForce uses.
    force_gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
    // PRIMARY mom flags.  Without overwrite_mom=1, QUDA's cudaMom is created
    // with QUDA_COPY_FIELD_CREATE from cpuMom (i.e. our zero-initialized
    // host buf, but interpreted as bits → uninitialized device storage)
    // and computeCloverForceQuda accumulates -force into uninitialized
    // memory → 1e+274 garbage.  See interface_quda.cpp:4070.
    force_gauge_param.overwrite_mom     = 1;  // cudaMom = ZERO before += -F
    force_gauge_param.use_resident_mom  = 0;
    force_gauge_param.make_resident_mom = 0;
    force_gauge_param.return_result_mom = 1;  // cpuMom.copy(cudaMom) at end
    QudaGaugeParam &gauge_param = force_gauge_param;

    // Coefficients: PowerNegHalf.residues[k] are the rational coefficients.
    // computeCloverForceQuda multiplies internally by 2·dt·coeff·kappa²
    // (see milc_interface.cpp:2655).  We want force only (no integration
    // step), so dt = 1.0 and the residues go in directly.
    std::vector<double> coeff(Npole);
    for (int k = 0; k < Npole; ++k) coeff[k] = PowerNegHalf.residues[k];

    const double kappa = inv_param.kappa;
    // QUDA_FORCE_DBG_K2_ZERO=1 → kappa2=0 (kills Wilson-hop force_coeff,
    // keeps σ-Oprod-A via ferm_epsilon[0]).
    // QUDA_FORCE_DBG_CK_ZERO=1 → ck=0 (kills σ-Oprod completely).
    const double kappa2 = std::getenv("QUDA_FORCE_DBG_K2_ZERO") ? 0.0
                                                                : -kappa * kappa;
    const double ck = std::getenv("QUDA_FORCE_DBG_CK_ZERO") ? 0.0
                                                            : -inv_param.clover_csw * kappa / 8.0;

    // Need a flat host gauge buffer too — pass the same one that's loaded
    // in QUDA (resident).  We don't have direct access to it; QUDA reads
    // the resident gaugePrecise.  computeCloverForceQuda accepts gauge =
    // nullptr and uses the resident.  Let's pass nullptr.
    // (MILC passes a real buffer but it's unused if resident.)

    // p (second array of vectors): the function signature ignores it
    // ("void**" — see lib/interface_quda.cpp:4488 the second array param
    // is unused in the symmetric clover case).  Pass a dummy.
    std::vector<void *> p_ptrs(Npole, nullptr);

    // QUDA_FORCE_DBG_NVEC1 — fall back to per-pole nvector=1 loop (workaround
    // for the upstream nvector>1 bug in older QUDA, e.g. agrebe Dec 2023).
    // Default: single fused call, requires QUDA >= chroma's 2025-07 snapshot.
    if (std::getenv("QUDA_FORCE_DBG_NVEC1")) {
      bool dbg_notrace = std::getenv("QUDA_FORCE_DBG_NOTRACE") != nullptr;
      for (int k = 0; k < Npole; ++k) {
        gauge_param.overwrite_mom = (k == 0) ? 1 : 0;
        double mult = (k == 0 && !dbg_notrace) ? 1.0 : 0.0;
        void *xp = x_ptrs[k];
        void *pp = p_ptrs[k];
        double cf = coeff[k];
        computeCloverForceQuda(mom_buf.data(), 1.0, &xp, &pp, &cf, kappa2, ck,
                               1, mult, nullptr, &gauge_param, &inv_param);
      }
    } else if (std::getenv("QUDA_FORCE_DBG_TM")) {
      // computeTMCloverForceQuda — chroma's tested path.
      // Coefficient convention: feed coeff[i] s.t. force_coeff[i] = coeff[i]
      // (TM doesn't multiply by 2·dt·kappa²); ferm_epsilon = {kcsw·coeff,
      // kcsw·coeff/κ²} with kcsw = κ·csw/8.  detratio=0, twist_flavor=NO.
      // Pre-scale our residue by -2·κ² to mimic the non-TM force_coeff
      // (since TM passes coeff directly).
      std::vector<void *> x0_ptrs(Npole, nullptr);
      std::vector<double> tm_coeff(Npole);
      for (int k = 0; k < Npole; ++k)
        tm_coeff[k] = -2.0 * kappa * kappa * coeff[k];
      computeTMCloverForceQuda(mom_buf.data(),
                               x_ptrs.data(),
                               x0_ptrs.data(),
                               tm_coeff.data(),
                               Npole,
                               &gauge_param,
                               &inv_param,
                               /*detratio=*/0);
    } else {
      // multiplicity=0: skip QUDA's σ_μν·F_μν trace term — that's the
      // LogDet(M_oo) derivative (hep-lat/0112051), which Grid handles
      // SEPARATELY via the log-det monomial with parity Odd
      // (QCDLogDetCompactCloverEO{,QudaForce}Action; before 2026-10-02 the
      // drivers paired this class with the EVEN block, L189).
      // Including it here would double-count.  QUDA_FORCE_DBG_TRACE=1
      // re-enables for diagnostic tests.
      double mult = std::getenv("QUDA_FORCE_DBG_TRACE") ? 1.0 : 0.0;
      computeCloverForceQuda(mom_buf.data(),
                             /*dt=*/1.0,
                             x_ptrs.data(),
                             p_ptrs.data(),
                             coeff.data(),
                             kappa2,
                             ck,
                             Npole,
                             mult,
                             /*gauge=*/nullptr,
                             &gauge_param,
                             &inv_param);
    }
    inv_param.use_resident_solution = saved_use_resident;
    inv_param.dagger                = saved_dagger;
    inv_param.twist_flavor          = saved_twist_fl;
    inv_param.mu                    = saved_mu;
    inv_param.epsilon               = saved_epsilon;
    inv_param.preserve_source       = saved_preserve;
    inv_param.cuda_prec_sloppy            = saved_sl;
    inv_param.cuda_prec_refinement_sloppy = saved_rs;
    inv_param.cuda_prec_precondition      = saved_pc;
    inv_param.gamma_basis                 = saved_basis;
    {
      double mom_norm = 0.0;
      int n_nan = 0;
      for (size_t i = 0; i < mom_buf.size(); ++i) {
        if (std::isnan(mom_buf[i])) ++n_nan;
        else mom_norm += mom_buf[i] * mom_buf[i];
      }
      std::cout << GridLogMessage << "[QudaForce] mom_buf size=" << mom_buf.size()
                << " norm2(non-NaN)=" << mom_norm << " n_nan=" << n_nan
                << " first10=";
      for (int i = 0; i < 10; ++i) std::cout << mom_buf[i] << " ";
      std::cout << std::endl;
    }

    // Unpack QUDA's 10-real anti-Hermitian momentum format → 3×3 complex.
    // The on-host MILC mom buffer is in EO site order; per-link layout
    // (derived from QUDA's Reconstruct<11>::Pack at gauge_field_order.h:1149,
    // copied flat through MILCOrder<*,10> in copy_gauge_inc.cu):
    //     m[0,1] = Re/Im(M_01)
    //     m[2,3] = Re/Im(M_02)
    //     m[4,5] = Re/Im(M_12)
    //     m[6]   = Im(M_00)
    //     m[7]   = Im(M_11)
    //     m[8]   = Im(M_22)              ← stored explicitly, NOT traceless-derived
    //     m[9]   = 0 (pad)
    // M is anti-Hermitian: M_ji = −conj(M_ij), Re(diag)=0.
    Coordinate lc = ggrid->LocalDimensions();
    std::vector<std::vector<double>> dir_eo_18(4, std::vector<double>(18 * V));
    for (int mu = 0; mu < 4; ++mu) {
      double *dst = dir_eo_18[mu].data();
      for (int site = 0; site < V; ++site) {
        const double *m = &mom_buf[(site * 4 + mu) * MOM_RECON];
        double r01 = m[0], i01 = m[1];
        double r02 = m[2], i02 = m[3];
        double r12 = m[4], i12 = m[5];
        double a0  = m[6], a1  = m[7], a2 = m[8];
        double *d = &dst[site * 18];
        d[ 0] = 0.0;   d[ 1] = a0;
        d[ 2] = r01;   d[ 3] = i01;
        d[ 4] = r02;   d[ 5] = i02;
        d[ 6] = -r01;  d[ 7] = i01;
        d[ 8] = 0.0;   d[ 9] = a1;
        d[10] = r12;   d[11] = i12;
        d[12] = -r02;  d[13] = i02;
        d[14] = -r12;  d[15] = i12;
        d[16] = 0.0;   d[17] = a2;
      }
    }

    std::vector<std::vector<double>> dir_lex_18(4, std::vector<double>(18 * V));
    double *lex_ptrs[4];
    for (int mu = 0; mu < 4; ++mu) {
      Quda::eo_to_lex_permute(dir_eo_18[mu].data(), dir_lex_18[mu].data(),
                              V, 18, lc);
      lex_ptrs[mu] = dir_lex_18[mu].data();
    }
    Quda::lex_buffers_to_gauge(lex_ptrs, dSdU);
    // QUDA convention: mom_buf = -force.  Sign flip + magnitude scale.
    // With multiplicity=0 (no LogDet), scale -1/(8κ²) gives:
    //   |PathB_scaled|² ≈ |Ta(PathA)|² (within 2%)
    //   cos(Ta(PathA), B) = +0.897 (15% perpendicular component remains)
    //   FD ratio = 0.712 (parallel component projects to ~0.905,
    //                     remaining perp adds RNG-dependent noise)
    // The 10% perp mismatch hasn't been narrowed via gamma_basis,
    // ck sign, Wilson-vs-σ isolation, or per-parity decomposition.
    // It's a deep convention difference between Grid's
    // MpcDeriv+MeeDeriv+MooDeriv decomposition and QUDA's
    // computeCloverOprod+computeCloverSigmaOprod+cloverDerivative pipeline.
    const double quda_to_grid_factor = -1.0 / (8.0 * kappa * kappa);
    dSdU = quda_to_grid_factor * dSdU;
    std::cout << GridLogMessage
              << "[QudaForce] κ=" << kappa
              << " norm2(PathB dSdU pre-scale)=" << norm2(dSdU)
              << std::endl;
    if (path_a_compare) {
      // Project PathA via Ta (traceless anti-Hermitian) — that's the part of
      // ∂S/∂U the SU(3) integrator actually consumes; PathB is already
      // anti-Hermitian by construction (the Hermitian part of ∂S/∂U is just
      // gauge-redundant noise).  Compare Ta(A) vs B for the apples-to-apples.
      GaugeField TaA(ggrid);
      for (int mu = 0; mu < Nd; ++mu) {
        auto Amu = PeekIndex<LorentzIndex>(dSdU_pathA, mu);
        auto TaAmu = Ta(Amu);
        PokeIndex<LorentzIndex>(TaA, TaAmu, mu);
      }
      double n2A_full = norm2(dSdU_pathA);
      double n2A_Ta   = norm2(TaA);
      double n2B = norm2(dSdU);
      auto inner_full = innerProduct(dSdU_pathA, dSdU);
      auto inner_Ta   = innerProduct(TaA, dSdU);
      std::cout << GridLogMessage
                << "[QudaForce] |PathA|²(full)=" << n2A_full
                << " |Ta(PathA)|²=" << n2A_Ta
                << " |PathB|²=" << n2B
                << " ⟨A,B⟩(full)=" << real(inner_full)
                << " ⟨Ta(A),B⟩=" << real(inner_Ta)
                << " factor Ta(A)/B = " << real(inner_Ta)/n2B
                << " cos(Ta(A),B)=" << real(inner_Ta)/std::sqrt(n2A_Ta*n2B)
                << std::endl;
      // Per-parity decomposition: compute |·|² and ⟨A,B⟩ on even-only and
      // odd-only site subsets.  If the misalignment is concentrated on one
      // parity, that's a smoking gun for missing-odd-contribution etc.
      {
        using SG = LatticeGaugeField::vector_object::scalar_object;
        Coordinate lcc = ggrid->LocalDimensions();
        double n2A_e=0, n2A_o=0, n2B_e=0, n2B_o=0;
        double iAB_e=0, iAB_o=0;
        std::vector<SG> scTa(V), scBp(V);
        unvectorizeToLexOrdArray(scTa, TaA);
        unvectorizeToLexOrdArray(scBp, dSdU);
        const double *bTa = reinterpret_cast<const double *>(scTa.data());
        const double *bBp = reinterpret_cast<const double *>(scBp.data());
        for (int site = 0; site < V; ++site) {
          int q = site, par = 0;
          for (int d = 0; d < (int)lcc.size(); ++d) { par += q % lcc[d]; q /= lcc[d]; }
          par &= 1;
          double dA = 0, dB = 0, dAB = 0;
          for (int r = 0; r < 72; ++r) { // 4 dirs * 18 reals
            double a = bTa[72*site + r], b = bBp[72*site + r];
            dA += a*a; dB += b*b; dAB += a*b;
          }
          if (par == 0) { n2A_e += dA; n2B_e += dB; iAB_e += dAB; }
          else          { n2A_o += dA; n2B_o += dB; iAB_o += dAB; }
        }
        std::cout << GridLogMessage
                  << "[QudaForce] EVEN: |Ta(A)|²=" << n2A_e
                  << " |B|²=" << n2B_e
                  << " ⟨Ta(A),B⟩=" << iAB_e
                  << " cos=" << iAB_e/std::sqrt(n2A_e*n2B_e)
                  << " factor=" << iAB_e/n2B_e
                  << std::endl;
        std::cout << GridLogMessage
                  << "[QudaForce] ODD:  |Ta(A)|²=" << n2A_o
                  << " |B|²=" << n2B_o
                  << " ⟨Ta(A),B⟩=" << iAB_o
                  << " cos=" << iAB_o/std::sqrt(n2A_o*n2B_o)
                  << " factor=" << iAB_o/n2B_o
                  << std::endl;
      }
      // Dump Ta(PathA) and PathB for site 0 mu 0 — same projection used
      // for the cos comparison.  Look for structural patterns (sign on
      // particular elements, ratios that vary, etc.).
      using SiteGauge = LatticeGaugeField::vector_object::scalar_object;
      std::vector<SiteGauge> scA(V), scB(V);
      unvectorizeToLexOrdArray(scA, TaA);   // Ta projection, not full A
      unvectorizeToLexOrdArray(scB, dSdU);
      const double *bA = reinterpret_cast<const double *>(scA.data());
      const double *bB = reinterpret_cast<const double *>(scB.data());
      Coordinate lc2 = ggrid->LocalDimensions();
      auto coords = [&](int site){
        std::stringstream s; int q = site;
        for (int d = 0; d < (int)lc2.size(); ++d) {
          s << (q % lc2[d]); s << (d+1 < (int)lc2.size() ? "," : "");
          q /= lc2[d];
        }
        return s.str();
      };
      // Dump full 18-real matrix for site=0 mu=0 in both A and B.
      {
        std::cout << GridLogMessage << "[QudaForce] site=0 mu=0 PathA 18 reals:";
        for (int r = 0; r < 18; ++r) std::cout << " " << bA[r];
        std::cout << std::endl;
        std::cout << GridLogMessage << "[QudaForce] site=0 mu=0 PathB 18 reals:";
        for (int r = 0; r < 18; ++r) std::cout << " " << bB[r];
        std::cout << std::endl;
        std::cout << GridLogMessage << "[QudaForce] site=0 mu=0 A/B per-real:";
        for (int r = 0; r < 18; ++r) {
          double a = bA[r], b = bB[r];
          if (std::abs(b) > 1e-30) std::cout << " " << a/b;
          else std::cout << " inf";
        }
        std::cout << std::endl;
      }
      const int n_sites_dump = std::min(4, V);
      for (int site = 0; site < n_sites_dump; ++site) {
        for (int mu = 0; mu < Nd; ++mu) {
          int q = site;
          int parity = 0;
          for (int d = 0; d < (int)lc2.size(); ++d) { parity += q % lc2[d]; q /= lc2[d]; }
          parity &= 1;
          double dotAA = 0.0, dotBB = 0.0, dotAB = 0.0;
          for (int r = 0; r < 18; ++r) {
            double a = bA[72*site + 18*mu + r];
            double b = bB[72*site + 18*mu + r];
            dotAA += a*a; dotBB += b*b; dotAB += a*b;
          }
          std::cout << GridLogMessage
                    << "[QudaForce] site=" << site << "(" << coords(site)
                    << ") parity=" << parity
                    << " mu=" << mu
                    << " |A|²=" << dotAA
                    << " |B|²=" << dotBB
                    << " ⟨A,B⟩=" << dotAB
                    << " B/A0=" << (std::abs(bA[72*site + 18*mu]) > 1e-30
                                    ? bB[72*site + 18*mu]/bA[72*site + 18*mu] : 0.0)
                    << std::endl;
        }
      }
      dSdU = dSdU_pathA;  // use PathA's correct force in compare mode
      return;
    }
  }

 private:
  QudaCloverParams qp_;
  std::unique_ptr<QudaCloverMultiShiftInverter> quda_ms_;
};

}  // namespace Grid
