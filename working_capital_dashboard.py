import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(layout="wide", page_title="Working Capital: Torre de Control Integral")

# ESTILOS MEJORADOS
st.markdown("""
    <style>
    .metric-card {background-color: #0e1117; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .big-font {font-size:24px !important; font-weight: bold;}
    .explanation {font-size: 14px; color: #888;}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS (BLINDADA)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS (BLINDADA V2 - SMART SEARCH)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS (BLINDADA V3 - ANTI-DUPLICADOS)
# -----------------------------------------------------------------------------
def load_data(uploaded_file):
    try:
        # A. LEER EL ARCHIVO SIN CABECERAS PARA BUSCAR DÓNDE EMPIEZA
        if uploaded_file.name.endswith('.csv'):
            df_temp = pd.read_csv(uploaded_file, header=None)
        else:
            df_temp = pd.read_excel(uploaded_file, header=None)

        # B. BUSCADOR INTELIGENTE DE CABECERA
        header_row = 0
        keywords = ['ventas', 'clientes', 'coste', 'ingresos', 'fecha', 'periodo']
        
        # Escaneamos las primeras 10 filas
        for i, row in df_temp.head(10).iterrows():
            # Convertimos la fila a una lista de textos simples para buscar
            row_text = [str(val).lower() for val in row.values]
            matches = sum(1 for x in row_text if any(k in x for k in keywords))
            if matches >= 2:
                header_row = i
                break
        
        # C. RECARGAR CON LA CABECERA CORRECTA
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=header_row)
        else:
            df = pd.read_excel(uploaded_file, header=header_row)

        # --- CORRECCIÓN DEL ERROR: ELIMINAR DUPLICADOS ---
        # Si hay dos columnas con el mismo nombre, nos quedamos solo con la primera
        df = df.loc[:, ~df.columns.duplicated()]
        
        # D. NORMALIZACIÓN DE NOMBRES (LIST COMPREHENSION - MÁS SEGURO)
        # Esto evita el error de .str en índices raros
        df.columns = [str(c).lower().strip().replace(' ', '_').replace('.', '') for c in df.columns]

        # Mapa de renombramiento
        rename_map = {
            'date': 'fecha', 'periodo': 'fecha', 'mes': 'fecha',
            'ventas': 'ventas_netas', 'ingresos': 'ventas_netas', 'facturacion': 'ventas_netas',
            'coste': 'coste_ventas', 'costos': 'coste_ventas', 'coste_ventas': 'coste_ventas', 'compras': 'coste_ventas',
            'clientes': 'cuentas_por_cobrar', 'deudores': 'cuentas_por_cobrar', 'cxc': 'cuentas_por_cobrar',
            'existencias': 'inventario', 'stock': 'inventario', 'inventarios': 'inventario',
            'proveedores': 'cuentas_por_pagar', 'acreedores': 'cuentas_por_pagar', 'cxp': 'cuentas_por_pagar'
        }
        
        new_columns = {}
        for col in df.columns:
            for key, val in rename_map.items():
                if key == col: 
                    new_columns[col] = val
                    break
        df.rename(columns=new_columns, inplace=True)
        
        # Volvemos a eliminar duplicados por si el renombre creó alguno nuevo
        df = df.loc[:, ~df.columns.duplicated()]

        # E. TRATAMIENTO DE FECHAS
        if 'fecha' not in df.columns:
            df.rename(columns={df.columns[0]: 'fecha'}, inplace=True)
            
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        if df['fecha'].isna().all() or df['fecha'].isnull().sum() > len(df) * 0.5:
             dates = pd.date_range(start='2024-01-01', periods=len(df), freq='ME')
             df['fecha'] = dates

        # F. LIMPIEZA NUMÉRICA (PROTEGIDA)
        cols_necesarias = ['ventas_netas', 'coste_ventas', 'cuentas_por_cobrar', 'inventario', 'cuentas_por_pagar']
        
        for col in cols_necesarias:
            if col not in df.columns:
                df[col] = 0.0 
            else:
                # Nos aseguramos de que sea una SERIE y no un DataFrame
                if isinstance(df[col], pd.DataFrame):
                    series = df[col].iloc[:, 0].astype(str) # Si sigue habiendo duplicados, coge el primero
                else:
                    series = df[col].astype(str)
                
                # Limpieza de caracteres
                series = series.str.replace(r'[€$a-zA-Z]', '', regex=True).str.strip()
                
                # Lógica Europea (1.000,00 -> 1000.00)
                # Si detectamos coma, asumimos que es decimal
                if series.str.contains(',', regex=False).any():
                    series = series.str.replace('.', '', regex=False) # Fuera punto de miles
                    series = series.str.replace(',', '.', regex=False) # Coma a punto
                
                df[col] = pd.to_numeric(series, errors='coerce').fillna(0.0)

        # DEBUG VISUAL
        with st.expander("✅ DATOS CARGADOS CORRECTAMENTE (CLICK PARA VER)", expanded=False):
            st.dataframe(df.head())

        return df

    except Exception as e:
        st.error(f"❌ Error crítico cargando datos: {e}")
        return None
# -----------------------------------------------------------------------------
# 2. GENERADOR DE DATOS (Simulación)
# -----------------------------------------------------------------------------
def generate_monthly_dummy_data(months=24):
    dates = pd.date_range(start="2023-01-01", periods=months, freq="ME")
    base_sales = 100000
    trend = np.linspace(1, 1.2, months)
    seasonality = 1 + 0.2 * np.sin(np.linspace(0, 4*np.pi, months))
    ventas = base_sales * trend * seasonality
    coste_ventas = ventas * 0.60
    # Simulamos ineficiencia (Gap) para que el gráfico se vea interesante
    clientes = (ventas / 30) * 55 
    inventario = (coste_ventas / 30) * 70 
    delta_inv = np.diff(inventario, prepend=inventario[0])
    compras = coste_ventas + delta_inv
    proveedores = (compras / 30) * 40 
    df = pd.DataFrame({
        'fecha': dates, 'ventas_netas': ventas, 'coste_ventas': coste_ventas,
        'compras': compras, 'cuentas_por_cobrar': clientes,
        'inventario': inventario, 'cuentas_por_pagar': proveedores
    })
    return df

# -----------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO
# -----------------------------------------------------------------------------
def calculate_financials(df, target_dso, target_dio, target_dpo):
    df = df.sort_values('fecha').reset_index(drop=True)
    dias_periodo = 30
    
    # A. Operativos
    df['dso'] = np.where(df['ventas_netas'] > 1, (df['cuentas_por_cobrar'] / df['ventas_netas']) * dias_periodo, 0)
    df['dio'] = np.where(df['coste_ventas'] > 1, (df['inventario'] / df['coste_ventas']) * dias_periodo, 0)
    
    base_dpo = df['compras'] if ('compras' in df.columns and df['compras'].sum() > 0) else df['coste_ventas']
    df['dpo'] = np.where(base_dpo > 1, (df['cuentas_por_pagar'] / base_dpo) * dias_periodo, 0)
    
    df['ccc'] = df['dso'] + df['dio'] - df['dpo']
    
    # Medias Móviles
    df['dso_trend'] = df['dso'].rolling(window=3, min_periods=1).mean()
    df['dio_trend'] = df['dio'].rolling(window=3, min_periods=1).mean()
    df['dpo_trend'] = df['dpo'].rolling(window=3, min_periods=1).mean()
    df['ccc_trend'] = df['ccc'].rolling(window=3, min_periods=1).mean()

    # B. Estratégicos
    df['nof_real'] = df['cuentas_por_cobrar'] + df['inventario'] - df['cuentas_por_pagar']
    
    df['ideal_clientes'] = (df['ventas_netas'] / dias_periodo) * target_dso
    df['ideal_inventario'] = (df['coste_ventas'] / dias_periodo) * target_dio
    df['ideal_proveedores'] = (base_dpo / dias_periodo) * target_dpo
    
    df['nof_ideal'] = df['ideal_clientes'] + df['ideal_inventario'] - df['ideal_proveedores']
    df['cash_gap'] = df['nof_real'] - df['nof_ideal']
    
    return df

# -----------------------------------------------------------------------------
# 4. LÓGICA DE INTERPRETACIÓN AVANZADA
# -----------------------------------------------------------------------------
def analyze_ccc_health(df, industry_type='manufacturing'):
    """
    Analiza la salud del Ciclo de Caja con reglas de negocio expertas (V3.0 - CFO Level).
    Devuelve una lista de alertas y el impacto económico (Cash Unlocked).
    """
    current = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else current
    avg = df.tail(12).mean(numeric_only=True)
    
    alerts = []
    
    # R1: CCC Negativo (Check de Industria)
    # Si es retail, CCC negativo es bueno (Amazon model), si es manufacturing, es riesgo.
    if current['ccc'] < 0 and industry_type != 'retail':
        alerts.append("🔴 **ALERTA CRÍTICA**: CCC Negativo en sector no-retail. Verificar si es eficiencia o incapacidad de pago (DPO forzado).")
        
    # R2: Mejora de CCC "mala" (por rotura de stock crítica)
    # Si el CCC baja (mejora) PERO el inventario cae drásticamente, es peligroso.
    delta_ccc = prev['ccc'] - current['ccc'] # Positivo = Mejora (menos días)
    inventory_drop = (current['inventario'] - prev['inventario']) / prev['inventario'] if prev['inventario'] > 0 else 0
    
    if delta_ccc > 5 and inventory_drop < -0.15: # Mejora de 5 días pero caída de 15% stock
        alerts.append(f"🟠 **ALERTA OPERATIVA**: Mejora de CCC artificial por caída de stock ({inventory_drop:.1%}). Riesgo inminente de rotura de stock (Stockouts).")
        
    # R3: DPO Extendido (Riesgo Proveedores)
    # Si el DPO sube más de un 20% vs la media histórica.
    if current['dpo'] > (avg['dpo'] * 1.20):
        alerts.append(f"🟡 **ALERTA PROVEEDORES**: DPO extendido (+{current['dpo'] - avg['dpo']:.1f} días vs media). Riesgo de tensión en cadena de suministro.")
        
    # R4: Cálculo de Impacto en Valoración (Cash Unlocked)
    # Cuánta caja se ha liberado (o consumido) al cambiar el CCC respecto al mes anterior
    # Fórmula: (CCC_anterior - CCC_actual) * Ventas_diarias
    ventas_diarias = current['ventas_netas'] / 30
    cash_unlocked = delta_ccc * ventas_diarias
    
    return alerts, cash_unlocked

# -----------------------------------------------------------------------------
# INTERFAZ DE USUARIO MEJORADA
# -----------------------------------------------------------------------------
st.title("📊 Working Capital: Torre de Control Integral")

# SIDEBAR
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu Excel Mensual", type=['xlsx', 'csv'])
st.sidebar.markdown("---")
st.sidebar.header("2. Simulador de Objetivos")
target_dso = st.sidebar.slider("Objetivo Cobro (DSO)", 15, 90, 30)
target_dio = st.sidebar.slider("Objetivo Stock (DIO)", 15, 120, 45)
target_dpo = st.sidebar.slider("Objetivo Pago (DPO)", 15, 120, 60)

# OPCIÓN DE FILTRO DE FECHA (NUEVO)
st.sidebar.markdown("---")
show_full_history = st.sidebar.checkbox("Mostrar todo el histórico", value=False)


if uploaded_file:
    df_raw = load_data(uploaded_file)
    if df_raw is not None:
        df = calculate_financials(df_raw, target_dso, target_dio, target_dpo)
        st.sidebar.success("✅ Datos Procesados")
    else:
        df = calculate_financials(generate_monthly_dummy_data(), target_dso, target_dio, target_dpo)
else:
    df = calculate_financials(generate_monthly_dummy_data(), target_dso, target_dio, target_dpo)

# --- FILTRADO DE DATOS (LÓGICA DE LOS 12 MESES) ---
if not show_full_history:
    df_display = df.tail(12).copy()
else:
    df_display = df.copy()

# ==============================================================================
# BLOQUE 1: DIAGNÓSTICO CLARO (KPIs)
# ==============================================================================
st.header("1. Diagnóstico Operativo (Último Cierre)")
st.markdown("Comparativa del mes actual frente al **promedio de tus últimos 12 meses**.")

last_month = df.iloc[-1]
avg_12m = df.tail(12).mean(numeric_only=True)

c1, c2, c3, c4 = st.columns(4)

def kpi_card_explained(col, title, value, avg, inverse=True, explanation=""):
    delta = value - avg
    is_good = delta < 0 if inverse else delta > 0
    delta_color_str = "normal" if is_good else "inverse"
    
    col.metric(
        label=title,
        value=f"{value:.1f} días",
        delta=f"{delta:.1f} vs Media ({avg:.1f})",
        delta_color=delta_color_str
    )
    col.caption(explanation)

kpi_card_explained(c1, "Ciclo de Caja (CCC)", last_month['ccc'], avg_12m['ccc'], True, "👇 Bajar es bueno.")
kpi_card_explained(c2, "DSO (Cobro)", last_month['dso'], avg_12m['dso'], True, "👇 Si baja, cobras antes.")
kpi_card_explained(c3, "DIO (Inventario)", last_month['dio'], avg_12m['dio'], True, "👇 Si baja, rotas más.")
kpi_card_explained(c4, "DPO (Pago)", last_month['dpo'], avg_12m['dpo'], False, "👆 Si sube, te financias.")

st.markdown("---")

# ==============================================================================
# BLOQUE 1.5: DIAGNÓSTICO IA AVANZADO (NUEVO)
# ==============================================================================
# Selector de industria para contexto
industry = st.sidebar.selectbox("Sector / Industria", ["manufacturing", "retail", "services", "tech"], index=0)

alerts, cash_impact = analyze_ccc_health(df, industry_type=industry)

st.subheader("🤖 Diagnóstico Inteligente de Salud Financiera")

# Mostrar impacto económico
col_ia1, col_ia2 = st.columns([1, 2])

with col_ia1:
    if cash_impact > 0:
        st.success(f"💰 **Caja Liberada (Mes vs Mes):** +${cash_impact:,.0f}")
        st.caption("La mejora en el ciclo ha generado liquidez extra.")
    elif cash_impact < 0:
        st.warning(f"💸 **Caja Atrapada (Mes vs Mes):** -${abs(cash_impact):,.0f}")
        st.caption("El deterioro del ciclo ha consumido caja operativa.")
    else:
        st.info("⚖️ **Caja Neutra:** El ciclo se mantiene estable.")

with col_ia2:
    if alerts:
        for alert in alerts:
            st.markdown(alert)
    else:
        st.success("✅ **Sistema Saludable**: No se detectan anomalías críticas en la estructura del capital de trabajo.")

st.markdown("---")

# ==============================================================================
# BLOQUE 1.8: VISUALIZACIÓN TÚNEL DE EFECTIVO (NUEVO)
# ==============================================================================
st.subheader("🕵️ Visualización del Túnel de Efectivo (Gantt)")
st.caption("Entiende visualmente tu brecha de financiación. La distancia entre el fin de los pagos (Rojo) y el fin de los cobros (Verde) es dinero que debes poner de tu bolsillo.")

# Preparación de Datos para Gráfico
# Usamos last_month que ya está definido arriba
dso_val = last_month['dso']
dio_val = last_month['dio']
dpo_val = last_month['dpo']
ccc_val = last_month['ccc']

fig_tunnel = go.Figure()

# --- CICLO OPERATIVO (FILA 1) ---
# 1. Inventario (Azul) - Empieza en 0
fig_tunnel.add_trace(go.Bar(
    y=['Ciclo Operativo'],
    x=[dio_val],
    name='Inventario (DIO)',
    orientation='h',
    marker=dict(color='#3498DB', line=dict(width=1)),
    hovertemplate="Inventario: %{x:.0f} días<extra></extra>"
))

# 2. Clientes (Verde) - Empieza después de Inventario 'base=dio_val'
# Nota: Plotly con barmode='stack' los apila automáticamente si están en la misma Y.
fig_tunnel.add_trace(go.Bar(
    y=['Ciclo Operativo'],
    x=[dso_val],
    name='Clientes (DSO)',
    orientation='h',
    marker=dict(color='#2ECC71', line=dict(width=1)),
    hovertemplate="Cobro: %{x:.0f} días<extra></extra>"
))

# --- FINANCIACIÓN (FILA 2) ---
# 3. Proveedores (Rojo) - Empieza en 0
fig_tunnel.add_trace(go.Bar(
    y=['Financiación'],
    x=[dpo_val],
    name='Proveedores (DPO)',
    orientation='h',
    marker=dict(color='#E74C3C', line=dict(width=1)), # Rojo sólido
    hovertemplate="Pago: %{x:.0f} días<extra></extra>"
))


# 4. Brecha (GAP) - Para rellenar visualmente la diferencia
# El GAP es la diferencia entre (DIO+DSO) y DPO.
total_operating_cycle = dio_val + dso_val
gap_size = total_operating_cycle - dpo_val

if gap_size > 0:
    fig_tunnel.add_trace(go.Bar(
        y=['Financiación'],
        x=[gap_size],
        base=[dpo_val], # Empieza donde acaba DPO
        name='NECESIDAD DE CAJA (CCC)',
        orientation='h',
        marker=dict(color='rgba(255, 165, 0, 0.4)', line=dict(color='#E67E22', width=2)), # Naranja semitransparente
        text=f"GAP: {gap_size:.0f} días",
        textposition='auto',
        hovertemplate="Dinero parado: %{x:.0f} días<extra></extra>"
    ))
elif gap_size < 0:
    # Caso de Cash Negativo (Financiación > Operativo)
    # Mostramos el excedente en la fila Operativo
    surplus = abs(gap_size)
    fig_tunnel.add_trace(go.Bar(
        y=['Ciclo Operativo'],
        x=[surplus],
        base=total_operating_cycle,
        name='EXCEDENTE DE CAJA',
        orientation='h',
        marker=dict(color='rgba(46, 204, 113, 0.4)', line=dict(color='#27AE60', width=2)), 
        text=f"Generando Caja: {surplus:.0f} días",
        textposition='auto',
        hovertemplate="Excedente: %{x:.0f} días<extra></extra>"
    ))


fig_tunnel.update_layout(
    title_text="Esquema Visual del Ciclo",
    barmode='stack', # Apilamos
    xaxis_title="Días Temporales (Desde compra MP)",
    yaxis=dict(autorange="reversed"), # Operativo arriba
    height=350,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
)

st.plotly_chart(fig_tunnel, use_container_width=True)

st.markdown("---")

# ==============================================================================
# BLOQUE 2: GRÁFICO (VENTAS vs ESTRÉS) - TEXTO RESTAURADO
# ==============================================================================
st.subheader("⚠️ Análisis de Estrés: ¿Morir de Éxito?")

# AQUÍ HEMOS RECUPERADO LA EXPLICACIÓN DETALLADA
with st.expander("ℹ️ ¿Cómo leer este gráfico? (Clic para abrir)", expanded=True):
    st.write("""
    Este gráfico cruza tus **Ventas (Barras Grises)** con tu **Ciclo de Caja (Línea Azul)**.
    * **Escenario Ideal:** Las barras suben (vendes más) y la línea baja o se mantiene plana (cobras eficiente).
    * **Escenario Peligroso ("Morir de Éxito"):** Las barras suben y la línea TAMBIÉN sube. Significa que vender más te está costando más dinero operativo (te estás ahogando en facturas pendientes y stock).
    """)

fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
# Barras: Ventas
fig_combo.add_trace(go.Bar(x=df_display['fecha'], y=df_display['ventas_netas'], name="Ventas (€)", marker_color='#D5DBDB', opacity=0.7), secondary_y=False)
# Línea: CCC
line_color = '#E74C3C' if last_month['ccc'] > 60 else '#2E86C1'
fig_combo.add_trace(go.Scatter(x=df_display['fecha'], y=df_display['ccc_trend'], name="Días de Caja (CCC)", line=dict(color=line_color, width=4), mode='lines+markers'), secondary_y=True)

fig_combo.update_layout(title_text="Correlación: Ventas vs Días de Caja (Últimos 12 meses)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig_combo, use_container_width=True)

st.markdown("---")

# ==============================================================================
# BLOQUE 3: ESTRATEGIA Y GAP
# ==============================================================================
st.header("2. Optimización: ¿Cuánto dinero hay sobre la mesa?")

gap_val = last_month['cash_gap']
col_a, col_b, col_c = st.columns([1, 1, 2])

col_a.metric("NOF Reales (Hoy)", f"{last_month['nof_real']:,.0f} €")
col_b.metric("NOF Óptimas (Meta)", f"{last_month['nof_ideal']:,.0f} €")
col_c.metric("💸 OPORTUNIDAD DE CAJA", f"{gap_val:,.0f} €", f"{'Ineficiencia' if gap_val > 0 else 'Ahorro'}", delta_color="inverse")

# GRÁFICO DE ÁREA
fig_area = go.Figure()
fig_area.add_trace(go.Scatter(x=df_display['fecha'], y=df_display['nof_real'], name='Realidad (NOF)', fill='tozeroy', line=dict(color='#E74C3C', width=2)))
fig_area.add_trace(go.Scatter(x=df_display['fecha'], y=df_display['nof_ideal'], name='Objetivo (Ideal)', fill='tozeroy', line=dict(color='#2ECC71', width=2)))
fig_area.update_layout(title_text="Visualización del Gap (Dinero Perdido)", height=400)
st.plotly_chart(fig_area, use_container_width=True)

# TABLA DESGLOSE
st.subheader("🔎 Detalle por Partida (Último Mes)")
breakdown_data = {
    'Partida': ['Clientes', 'Inventario', 'Proveedores'],
    'Días Reales': [last_month['dso'], last_month['dio'], last_month['dpo']],
    'Días Objetivo': [target_dso, target_dio, target_dpo],
    'Saldo Real (€)': [last_month['cuentas_por_cobrar'], last_month['inventario'], last_month['cuentas_por_pagar']],
    'Saldo Óptimo (€)': [last_month['ideal_clientes'], last_month['ideal_inventario'], last_month['ideal_proveedores']],
}
df_breakdown = pd.DataFrame(breakdown_data)
df_breakdown['Diferencia (€)'] = 0.0
df_breakdown.loc[0, 'Diferencia (€)'] = df_breakdown.loc[0, 'Saldo Real (€)'] - df_breakdown.loc[0, 'Saldo Óptimo (€)']
df_breakdown.loc[1, 'Diferencia (€)'] = df_breakdown.loc[1, 'Saldo Real (€)'] - df_breakdown.loc[1, 'Saldo Óptimo (€)']
df_breakdown.loc[2, 'Diferencia (€)'] = (df_breakdown.loc[2, 'Saldo Óptimo (€)'] - df_breakdown.loc[2, 'Saldo Real (€)']) 

st.dataframe(df_breakdown, column_config={"Saldo Real (€)": st.column_config.NumberColumn(format="%.0f €"), "Saldo Óptimo (€)": st.column_config.NumberColumn(format="%.0f €"), "Diferencia (€)": st.column_config.NumberColumn(format="%.0f €"), "Días Reales": st.column_config.NumberColumn(format="%.1f d")}, use_container_width=True, hide_index=True)

# ==============================================================================
# BLOQUE 4: TABLA DE HISTÓRICO
# ==============================================================================
st.subheader(f"3. Histórico de Datos ({'Completo' if show_full_history else 'Últimos 12 Meses'})")

cols_to_show = ['fecha', 'ventas_netas', 'coste_ventas', 'inventario', 'cuentas_por_cobrar', 'cuentas_por_pagar', 'ccc', 'dso', 'dio', 'dpo']

st.dataframe(
    df_display[cols_to_show].sort_values('fecha', ascending=False), 
    column_config={
        "fecha": st.column_config.DateColumn(format="DD/MM/YYYY"), 
        "ventas_netas": st.column_config.NumberColumn(format="%.0f €"),
        "coste_ventas": st.column_config.NumberColumn(format="%.0f €"),
        "inventario": st.column_config.NumberColumn(format="%.0f €"),
        "cuentas_por_cobrar": st.column_config.NumberColumn(format="%.0f €"),
        "cuentas_por_pagar": st.column_config.NumberColumn(format="%.0f €"),
    }, 
    use_container_width=True, 
    hide_index=True
)