package front

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"stefan-service/internal/erfx"
	"stefan-service/internal/stefan"
)

func iceParams() stefan.Params {
	return stefan.Params{C: 2100, Lf: 334e3, K: 2.22, Alpha: 1.15e-6, Tf: 0, Tw: -20}
}

// TestZeroTimeFront 时刻为零，锋面位置恒为零（热流按理论为 +∞，HTTP 层转 null）。
func TestZeroTimeFront(t *testing.T) {
	r, err := Compute(iceParams(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if r.Front != 0 {
		t.Errorf("t=0 锋面位置 = %v，要求恒为 0", r.Front)
	}
	if !math.IsInf(r.WallHeatFlux, 1) || !math.IsInf(r.FrontSpeed, 1) {
		t.Errorf("t=0 理想阶跃边界下热流/速度应为 +Inf")
	}
}

// TestQuarterTimeRootRelation 时间放大 4 倍，锋面位置正好加倍（√t 律）。
func TestQuarterTimeRootRelation(t *testing.T) {
	p := iceParams()
	r1, err := Compute(p, 900) // 15 min
	if err != nil {
		t.Fatal(err)
	}
	r4, err := Compute(p, 3600) // 60 min = 4×
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(r4.Front-2*r1.Front)/(2*r1.Front) > 1e-12 {
		t.Errorf("s(4t)=%.9f 与 2s(t)=%.9f 不符", r4.Front, 2*r1.Front)
	}
	// 同一工况两次核算 λ 必须一致。
	if r1.Lambda.Lambda != r4.Lambda.Lambda {
		t.Errorf("同工况 λ 不一致：%.15f vs %.15f", r1.Lambda.Lambda, r4.Lambda.Lambda)
	}
}

// TestDeeperWithLargerSubcooling 只加大壁面与凝固点温差，同一时刻锋面更深。
func TestDeeperWithLargerSubcooling(t *testing.T) {
	base := iceParams()
	colder := iceParams()
	colder.Tw = -40
	tm := 3600.0
	ra, err := Compute(base, tm)
	if err != nil {
		t.Fatal(err)
	}
	rb, err := Compute(colder, tm)
	if err != nil {
		t.Fatal(err)
	}
	if !(rb.Front > ra.Front) {
		t.Errorf("更大温差反而推进更浅：%.6f <= %.6f", rb.Front, ra.Front)
	}
	if !(rb.WallHeatFlux > ra.WallHeatFlux) {
		t.Errorf("更大温差壁面热流应更大")
	}
}

// TestStagnationAsLatentHeatDiverges 潜热趋于很大（Ste→0），锋面趋于停滞。
func TestStagnationAsLatentHeatDiverges(t *testing.T) {
	reference := Compute
	_ = reference
	tm := 3600.0
	prev := math.Inf(1)
	for _, lf := range []float64{334e3, 3.34e6, 3.34e9, 3.34e12} {
		p := iceParams()
		p.Lf = lf
		r, err := Compute(p, tm)
		if err != nil {
			t.Fatal(err)
		}
		if r.Ste <= 0 || r.Front <= 0 {
			t.Fatalf("物理量应为正：Ste=%v s=%v", r.Ste, r.Front)
		}
		if r.Front >= prev {
			t.Errorf("潜热加大锋面应收窄：Lf=%g 时 s=%.3e >= 上一档 %.3e", lf, r.Front, prev)
		}
		prev = r.Front
	}
	// 极限档：Ste≈6.3e-9，一小时推进应小于 1 微米。
	p := iceParams()
	p.Lf = 3.34e15
	r, _ := Compute(p, tm)
	if r.Front > 1e-6 {
		t.Errorf("Ste 趋于零时锋面未停滞：s=%.3e", r.Front)
	}
}

// TestHeatFluxStefanBalance 壁面热流必须满足全局能量守恒自洽。
//
// 界面 Stefan 条件：ρLf·ds/dt = k·∂T/∂x|_x=s = q_w·e^{-λ²}。
// 壁面取热除支付界面潜热外，还要持续给不断加厚、整体降温的固相
// 提供显热，故 q_w > ρLf·ds/dt，二者相差显热累计速率。
// 固相体积内能（相对 Tf），η=x/(2√αt)：
//
//	U_s = ρc∫_0^s (T(x)-Tf) dx
//	    = -2ρc(Tf-Tw)√(αt) · (1-e^{-λ²})/(√π·erf(λ))
//
// （积分用 ∫_0^λ erf(η)dη = λ·erf(λ)+(e^{-λ²}−1)/√π 化简。）
// 全局账：q_w = ρLf·ds/dt - dU_s/dt（U_s<0，故第二项为正）。
// 以前这条测试错误地断言界面潜热流等于壁面热流（相当于无视固相显热，
// 恰与壁面公式误乘 e^{-λ²} 的缺陷相互掩盖），已改为正确的全局闭合。
func TestHeatFluxStefanBalance(t *testing.T) {
	p := iceParams()
	r, err := Compute(p, 1800)
	if err != nil {
		t.Fatal(err)
	}
	lam := r.Lambda.Lambda
	rho := p.K / (p.Alpha * p.C) // 由 α=k/(ρc) 反演密度
	latent := rho * p.Lf * r.FrontSpeed

	// 1) 界面 Stefan 条件：潜热流 = 界面处传导热流 = q_w·e^{-λ²}。
	interfaceFlux := r.WallHeatFlux * math.Exp(-lam*lam)
	if math.Abs(interfaceFlux-latent)/latent > 1e-9 {
		t.Errorf("界面 Stefan 条件不闭合：ρLf·ds/dt %.6f vs q_w·e^{-λ²} %.6f",
			latent, interfaceFlux)
	}

	// 2) 全局能量闭合：q_w = 潜热流 + 固相显热累计速率（-dU_s/dt）。
	sensible := rho * p.C * (p.Tf - p.Tw) * math.Sqrt(p.Alpha/r.Time) *
		(1 - math.Exp(-lam*lam)) / (math.SqrtPi * erfx.Erf(lam))
	if math.Abs(r.WallHeatFlux-(latent+sensible))/r.WallHeatFlux > 1e-9 {
		t.Errorf("全局能量不闭合：壁面热流 %.6f vs 潜热 %.6f + 显热 %.6f",
			r.WallHeatFlux, latent, sensible)
	}
	// 3) 固相在持续降温，壁面热流必须严格大于纯潜热流。
	if !(sensible > 0 && r.WallHeatFlux > latent && r.FrontSpeed > 0) {
		t.Errorf("显热应为正且 q_w > 潜热流：sensible=%.6f", sensible)
	}
	// 4) 显热/潜热 = e^{λ²}−1（Stefan 方程代入即得），作为旁证再钉一次。
	ratio := sensible / latent
	if math.Abs(ratio-(math.Exp(lam*lam)-1)) > 1e-9 {
		t.Errorf("显热/潜热 = %.9f，预期 e^{λ²}−1 = %.9f",
			ratio, math.Exp(lam*lam)-1)
	}
}

// TestIceOneHourCentimeters 预置冰层算例：1 小时凝固厚度为厘米量级（约 3.16 cm）。
func TestIceOneHourCentimeters(t *testing.T) {
	r, err := Compute(iceParams(), 3600)
	if err != nil {
		t.Fatal(err)
	}
	if r.Front < 0.02 || r.Front > 0.05 {
		t.Errorf("冰层 1 小时厚度 = %.4f m，不在预期厘米量级区间", r.Front)
	}
	if math.Abs(r.Front-0.0316) > 0.002 {
		t.Errorf("冰层 1 小时厚度 = %.5f m，预期约 0.0316 m", r.Front)
	}
}

// TestNegativeTime 负时刻必须被拒。
func TestNegativeTime(t *testing.T) {
	if _, err := Compute(iceParams(), -1); !errors.Is(err, ErrNegativeTime) {
		t.Fatalf("负时刻未被拦截：%v", err)
	}
}

// TestSeriesCancellation 取消后绝不能返回半成品点列。
func TestSeriesCancellation(t *testing.T) {
	spec := SeriesSpec{Params: iceParams(), T0: 0, T1: 86400, Intervals: 200_000_000}

	// 1) 启动前即取消：直接报错，零值结果。
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ComputeSeries(ctx, spec); !errors.Is(err, context.Canceled) {
		t.Fatalf("预取消应返回 context.Canceled，得到 %v", err)
	}

	// 2) 运行中取消：同样必须报错且不携带点列。
	ctx2, cancel2 := context.WithCancel(context.Background())
	done := make(chan struct{})
	var res SeriesResult
	var err2 error
	go func() {
		res, err2 = ComputeSeries(ctx2, spec)
		close(done)
	}()
	cancel2()
	<-done
	if !errors.Is(err2, context.Canceled) {
		t.Fatalf("运行中取消应返回 context.Canceled，得到 %v", err2)
	}
	if res.Points != nil || res.Lambda != 0 {
		t.Fatalf("取消后泄露了半成品结果：%+v", res)
	}
}

// TestSeriesHappyPath 正常序列满足 √t 律且点数完整。
func TestSeriesHappyPath(t *testing.T) {
	spec := SeriesSpec{Params: iceParams(), T0: 0, T1: 3600, Intervals: 4}
	res, err := ComputeSeries(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Points) != 5 {
		t.Fatalf("采样点数 = %d，要求 5", len(res.Points))
	}
	if res.Points[0].Front != 0 {
		t.Errorf("首点 t=0 锋面应为 0")
	}
	if math.Abs(res.Points[4].Front-2*res.Points[1].Front)/res.Points[4].Front > 1e-12 {
		t.Errorf("序列内 √t 关系不成立")
	}
}

// TestSeriesValidation 非法参数在作业计算前被拦下。
func TestSeriesValidation(t *testing.T) {
	bad := iceParams()
	bad.Alpha = 0
	_, err := ComputeSeries(context.Background(), SeriesSpec{
		Params: bad, T0: 0, T1: 10, Intervals: 2,
	})
	var ve *stefan.ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("期望 *stefan.ValidationError，得到 %v", err)
	}
}

// TestWallHeatFluxExternalGradient 用完全独立于服务实现的外部推导钉死壁面热流。
//
// 该量曾被误乘界面处的 e^{-λ²}：内部潜热账依旧自洽（缺陷被掩盖），
// 只有回到温度场在 x=0 的真实梯度才显形。为防悄悄漂回，这里：
//
//  1. erf 不用 internal/erfx，改用自适应思路下的高分辨率 Simpson 直接积分
//     erf(z) = 2/√π ∫_0^z e^{-u²}du；
//  2. λ 不用 stefan.SolveLambda，对超越方程独立二分到机器精度；
//  3. 解析壁面梯度 q_w = k(Tf−Tw)/(√(παt))·1/erf(λ) 给出独立参考值；
//  4. 再对 Neumann 温度场在壁面做中心差分（h 与 h/2 两步 Richardson
//     外推，O(h⁴)），不依赖任何解析梯度公式，三方必须一致。
func TestWallHeatFluxExternalGradient(t *testing.T) {
	p := iceParams()
	const tm = 3600.0
	r, err := Compute(p, tm)
	if err != nil {
		t.Fatal(err)
	}

	// --- 外部推导：独立 erf（Simpson 积分定义式） ---
	erfIndependent := func(z float64) float64 {
		const n = 20001 // 奇数点 → 偶数区间；积分误差对 λ≈0.25 远小于 1e-12
		h := z / float64(n-1)
		sum := 1.0 + math.Exp(-z*z)
		for i := 1; i < n-1; i++ {
			u := h * float64(i)
			w := 4.0
			if i%2 == 0 {
				w = 2.0
			}
			sum += w * math.Exp(-u*u)
		}
		return 2 / math.SqrtPi * h / 3 * sum
	}

	// --- 外部推导：独立二分求 λ：√π λ e^{λ²} erf(λ) = Ste ---
	ste := p.C * (p.Tf - p.Tw) / p.Lf
	f := func(l float64) float64 {
		return math.SqrtPi * l * math.Exp(l*l) * erfIndependent(l)
	}
	lo, hi := 0.0, 1.0
	for f(hi) < ste {
		hi *= 2
	}
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if f(mid) < ste {
			lo = mid
		} else {
			hi = mid
		}
	}
	lam := (lo + hi) / 2

	// 外部推导的 λ 必须与服务一致（证明外部推导建立在同一个相似常数上）。
	if math.Abs(lam-r.Lambda.Lambda) > 1e-12 {
		t.Fatalf("外部独立 λ=%.10f 与服务 λ=%.10f 不一致", lam, r.Lambda.Lambda)
	}

	dT := p.Tf - p.Tw
	rootAT := math.Sqrt(p.Alpha * tm)
	qAnalytic := p.K * dT / (math.SqrtPi * rootAT) / erfIndependent(lam)

	// 预置冰层工况一小时的外部梯度参考值：约 1432.4 W/m²。
	if math.Abs(qAnalytic-1432.4)/1432.4 > 1e-4 {
		t.Fatalf("外部解析梯度参考值漂移：q=%.4f，预期约 1432.4 W/m²", qAnalytic)
	}

	// 服务输出必须与外部解析梯度一致到机器精度量级；
	// 旧实现（误乘 e^{-λ²}）会偏 ~6.2%，必然落网。
	if rel := math.Abs(r.WallHeatFlux-qAnalytic) / qAnalytic; rel > 1e-9 {
		t.Errorf("壁面热流 %.4f 与外部梯度推导 %.4f 相对偏差 %.2e，要求 < 1e-9",
			r.WallHeatFlux, qAnalytic, rel)
	}

	// --- 外部推导：直接对温度场有限差分求壁面梯度（不用解析梯度公式） ---
	temp := func(x float64) float64 {
		return p.Tw + dT*erfIndependent(x/(2*rootAT))/erfIndependent(lam)
	}
	gradient := func(h float64) float64 {
		return (temp(h) - temp(-h)) / (2 * h) // 壁面 x=0 中心差分
	}
	h := 1e-3 * r.Front // 远小于锋面尺度，但远超浮点舍入区
	g1 := gradient(h)
	g2 := gradient(h / 2)
	gRich := (16*g2 - g1) / 15 // Richardson 外推：主导误差 O(h²) → O(h⁴)
	qFD := p.K * gRich
	if math.Abs(g1-g2)/gRich > 1e-6 {
		t.Fatalf("有限差分外推未收敛：g(h)=%.8f g(h/2)=%.8f", g1, g2)
	}
	if rel := math.Abs(r.WallHeatFlux-qFD) / qFD; rel > 1e-6 {
		t.Errorf("壁面热流 %.4f 与温度场有限差分梯度 %.4f 相对偏差 %.2e，要求 < 1e-6",
			r.WallHeatFlux, qFD, rel)
	}

	// 旧的错误值必须明确出局，防止测试被写松。
	if math.Abs(r.WallHeatFlux-1348.47) < 5 {
		t.Errorf("壁面热流 %.4f 仍是旧的界面热流错误值（约 1348.47）", r.WallHeatFlux)
	}
}

// TestParallelIsolation 多组工况并行推算，各自 λ 与锋面互不渗透。
func TestParallelIsolation(t *testing.T) {
	mk := func(tw float64) stefan.Params {
		p := iceParams()
		p.Tw = tw
		return p
	}
	cases := []struct {
		p    stefan.Params
		t    float64
		sMin float64
		sMax float64
	}{
		{mk(-5), 3600, 0.014, 0.017},
		{mk(-20), 3600, 0.030, 0.034},
		{mk(-60), 1800, 0.035, 0.045},
	}
	// 先算单线程基准。
	want := make([]Result, len(cases))
	for i, c := range cases {
		r, err := Compute(c.p, c.t)
		if err != nil {
			t.Fatal(err)
		}
		want[i] = r
	}

	const goroutines = 32
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines*len(cases)*5)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for rep := 0; rep < 5; rep++ {
				for i, c := range cases {
					r, err := Compute(c.p, c.t)
					if err != nil {
						errCh <- err
						return
					}
					if r.Lambda.Lambda != want[i].Lambda.Lambda ||
						r.Front != want[i].Front ||
						r.Ste != want[i].Ste {
						errCh <- errors.New("并行工况结果与单线程基准不一致，存在状态渗透")
						return
					}
					if r.Front < c.sMin || r.Front > c.sMax {
						errCh <- errors.New("锋面落出该工况独有区间")
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}
