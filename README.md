# Retrofit
Input:
• B = {b₁, b₂, ..., bₙ}: Set of buildings with characteristics and energy consumption data
• M = {m₁, m₂, ..., mₖ}: Set of retrofit measures with associated parameters
• E = {r, p}: Economic parameters where r is discount rate and p is energy price
Output:
• R: Economic indicators (NPV, ROI, payback period) for each measure by cluster
Procedure:
1:	function CalculateRetrofitEconomics(B, M, E)
2:	R ← ∅ ▷ Initialize empty results collection
3:	r ← E.discount_rate ▷ Extract discount rate
4:	p ← E.energy_price ▷ Extract energy price
5:	C ← ClusterBuildings(B) ▷ Group buildings into clusters
6:	for each cluster c in C do
7:	Ec ← mean(energy_consumption(c)) ▷ Average energy consumption
8:	Ac ← mean(building_area(c)) ▷ Average building area
9:	Rc ← ∅ ▷ Initialize results for this cluster
10:	for each measure m in M do
11:	id ← m.identifier
12:	cost₀ ← m.cost_per_m² × Ac ▷ Initial investment cost
13:	η ← m.energy_saving_percentage/100 ▷ Energy saving efficiency
14:	S_energy ← Ec × η × Ac ▷ Annual energy savings (kWh)
15:	S_annual ← S_energy × p ▷ Annual monetary savings (€)
16:	L ← m.expected_lifespan ▷ Lifespan in years
17:	NPV ← -cost₀ ▷ Begin with negative investment
18:	for t ← 1 to L do
19:	NPV ← NPV + S_annual/((1+r)ᵗ) ▷ Add discounted annual savings
20:	end for
21:	ROI ← NPV/cost₀ ▷ Return on investment ratio
22:	PP ← cost₀/S_annual ▷ Payback period in years
23:	Rc[id] ← {NPV, ROI, PP, cost₀, S_annual, L}
24:	end for
25:	R[c] ← Rc
26:	end for
27:	return R
28:	end function
29:	function RankInterventionsByMetric(R, metric)
30:	T ← ∅ ▷ Initialize ranking result
31:	for each cluster c in R do
32:	if metric ∈ {NPV, ROI} then
33:	Tc ← sort(R[c], metric, descending) ▷ Higher values are better
34:	else
35:	Tc ← sort(R[c], metric, ascending) ▷ Lower values are better
36:	end if
37:	T[c] ← Tc
38:	end for
39:	return T
40:	end function
