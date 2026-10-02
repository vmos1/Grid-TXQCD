#pragma once
// EVEN-parity variant of TwoFlavourSchurCloverAction.
//
// NOT the identical action on its own (corrected 2026-10-02): det(M_pc_oo) = det M / det M_ee
// and det(M_pc_ee) = det M / det M_oo, and with a clover term det M_ee != det M_oo. Sampling
//   PhiEven ~ exp[-φ† (M_pc_ee†M_pc_ee)^(-1) φ] instead of
//   PhiOdd  ~ exp[-φ† (M_pc_oo†M_pc_oo)^(-1) φ]
// therefore requires the partner Nf=2 log-det monomial on the OTHER block, -2 ln|det M_oo|
// (QCDLogDet*CloverEOAction with parity Odd), not the even block (grid_qcd docs
// 2026_10_02_strange_logdet_parity_mismatch.md, L189). No production driver pairs this class
// with a log-det yet; whoever does must use parity Odd.
//
// Reason for the EVEN variant: QUDA's `computeCloverForceQuda` only
// supports / is tested on EVEN_EVEN_ASYMMETRIC matpc (PyQUDA exclusively
// uses this). Wiring an EVEN-parity Nf=2 action into HMC is the cleanest
// path to plug QUDA's force routine into TwoFlavour.
//
// Mirrors TwoFlavourSchurCloverAction line for line:
//   PhiOdd → PhiEven, pickCheckerboard(Odd → Even).
//   MooDeriv (diagonal clover on odd) → MeeDeriv (diagonal clover on even).
//   Meooe even→odd projection chain on the off-parity completion.
// Hopping derivatives MpcDeriv/MpcDagDeriv via SchurDifferentiableOperator
// are parity-agnostic — same kernel code applies on either checkerboard.

#include <Grid/qcd/action/fermion/WilsonCloverFermion.h>

NAMESPACE_BEGIN(Grid);

template <class Impl,
          class FermionOp = WilsonCloverFermion<Impl, Grid::CloverHelpers<Impl>>>
class TwoFlavourSchurCloverActionEven
    : public Action<typename Impl::GaugeField> {
public:
  INHERIT_IMPL_TYPES(Impl);

  typedef FermionOp FermionOperator;

protected:
  FermionOperator &FermOp;
  OperatorFunction<FermionField> &DerivativeSolver;
  OperatorFunction<FermionField> &ActionSolver;
  FermionField PhiEven;

public:
  TwoFlavourSchurCloverActionEven(FermionOperator &Op,
                                  OperatorFunction<FermionField> &DS,
                                  OperatorFunction<FermionField> &AS)
      : FermOp(Op), DerivativeSolver(DS), ActionSolver(AS),
        PhiEven(Op.FermionRedBlackGrid()) {}

  std::string action_name() override {
    return "TwoFlavourSchurCloverActionEven";
  }

  std::string LogParameters() override {
    std::stringstream os;
    os << GridLogMessage << "[" << action_name() << "] has no parameters" << std::endl;
    return os.str();
  }

  void refresh(const GaugeField &U, GridSerialRNG &sRNG,
               GridParallelRNG &pRNG) override {
    RealD scale = std::sqrt(0.5);

    FermionField eta(FermOp.FermionGrid());
    FermionField etaEven(FermOp.FermionRedBlackGrid());

    gaussian(pRNG, eta);
    pickCheckerboard(Even, etaEven, eta);

    FermOp.ImportGauge(U);
    SchurDifferentiableOperator<Impl> PCop(FermOp);

    PCop.MpcDag(etaEven, PhiEven);
    PhiEven = PhiEven * scale;
  }

  RealD S(const GaugeField &U) override {
    FermOp.ImportGauge(U);

    FermionField X(FermOp.FermionRedBlackGrid());
    FermionField Y(FermOp.FermionRedBlackGrid());

    SchurDifferentiableOperator<Impl> PCop(FermOp);

    X = Zero();
    ActionSolver(PCop, PhiEven, X);
    PCop.Op(X, Y);
    RealD action = norm2(Y);

    std::cout << GridLogMessage << "[" << action_name() << "] S = " << action
              << std::endl;
    return action;
  }

  void deriv(const GaugeField &U, GaugeField &dSdU) override {
    FermOp.ImportGauge(U);

    GridBase *fcbgrid = FermOp.FermionRedBlackGrid();
    FermionField X(fcbgrid);
    FermionField Y(fcbgrid);
    GaugeField tmp(FermOp.GaugeGrid());

    SchurDifferentiableOperator<Impl> Mpc(FermOp);

    // Solve for X = (Mpc†Mpc)^{-1} PhiEven, Y = Mpc X
    X = Zero();
    DerivativeSolver(Mpc, PhiEven, X);
    Mpc.Mpc(X, Y);

    // 1. Hopping forces (parity-agnostic in SchurDifferentiableOperator).
    Mpc.MpcDeriv(tmp, Y, X);
    dSdU = tmp;
    Mpc.MpcDagDeriv(tmp, X, Y);
    dSdU = dSdU + tmp;

    // 2. Even-site clover force (M_ee derivative — diagonal block on EVEN
    //    parity). Mirror of MooDeriv in the odd-parity version.
    FermOp.MeeDeriv(tmp, Y, X, DaggerNo);
    dSdU = dSdU + tmp;
    FermOp.MeeDeriv(tmp, X, Y, DaggerYes);
    dSdU = dSdU + tmp;

    // 3. Odd-site clover force (chain rule through Moo^{-1}).
    //    W_o = Moo^{-1} Meo X,  Z_o = Moo^{-†} Moe† Y
    FermionField W_o(fcbgrid), Z_o(fcbgrid), tmp1(fcbgrid);

    FermOp.Meooe(X, tmp1);          // even → odd
    FermOp.MooeeInv(tmp1, W_o);     // odd → odd

    FermOp.MeooeDag(Y, tmp1);       // even → odd
    FermOp.MooeeInvDag(tmp1, Z_o);  // odd → odd

    FermOp.MooDeriv(tmp, Z_o, W_o, DaggerNo);
    dSdU = dSdU + tmp;
    FermOp.MooDeriv(tmp, W_o, Z_o, DaggerYes);
    dSdU = dSdU + tmp;
  }
};

NAMESPACE_END(Grid);
