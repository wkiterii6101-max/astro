"""
천체 FITS 분석 앱
------------------------------------------------
- 업로드한 FITS 파일에서 스펙트럼(1차원 flux-wavelength 데이터)을 추출하여
  흑체복사(Planck 함수) 피팅으로 표면 온도를 계산합니다.
- FITS 헤더의 WCS 정보를 이용해 천체의 하늘 좌표(RA, Dec)를 계산합니다.
- Streamlit UI로 결과(스펙트럼 그래프, 온도, 좌표, 이미지+마커)를 시각화합니다.

실행: streamlit run app.py
필요 패키지: streamlit, astropy, numpy, scipy, matplotlib
"""

import io

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 그래프(matplotlib)에는 한글 폰트가 없는 환경(예: Streamlit Cloud)에서도
# 글자가 깨지지 않도록 축/제목/범례 텍스트를 전부 영어로 표기합니다.
# (Streamlit UI 쪽 한글 텍스트는 브라우저가 렌더링하므로 문제 없습니다.)
plt.rcParams["axes.unicode_minus"] = False

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.constants import h, c, k as k_B


# ------------------------------------------------------------------
# 1. 물리 상수 & 흑체복사(Planck) 함수
# ------------------------------------------------------------------
def planck_lambda(wavelength_m, T):
    """
    흑체복사 스펙트럼 세기 (단위 파장당 복사휘도).
    wavelength_m : 파장 [m] 단위의 배열
    T            : 온도 [K]
    """
    wavelength_m = np.asarray(wavelength_m, dtype=np.float64)
    exponent = (h * c) / (wavelength_m * k_B * T)
    # 오버플로 방지
    exponent = np.clip(exponent, -700, 700)
    intensity = (2 * h * c ** 2) / (wavelength_m ** 5) / (np.exp(exponent) - 1)
    return intensity


def wien_peak_temperature(wavelength_m, flux):
    """
    빈의 변위법칙(Wien's displacement law)을 이용한 초기 온도 추정값.
    b = 2.8977719e-3 m·K
    """
    b = 2.8977719e-3
    peak_idx = np.argmax(flux)
    lambda_peak = wavelength_m[peak_idx]
    if lambda_peak <= 0:
        return 5778.0  # 태양 온도를 기본값으로 사용
    return b / lambda_peak


# ------------------------------------------------------------------
# 2. FITS 파일에서 스펙트럼(파장, flux) 데이터 추출
# ------------------------------------------------------------------
def extract_spectrum(hdul):
    """
    FITS HDU 리스트에서 1차원 스펙트럼 데이터를 찾아 (wavelength[m], flux) 반환.
    - 1차원 데이터 + WCS(CRVAL1/CDELT1/CRPIX1)로 파장축을 만드는 경우를 우선 처리.
    - 파장 단위 헤더(CUNIT1)가 있으면 nm/Angstrom -> m 변환.
    """
    data = None
    header = None
    for hdu in hdul:
        if hdu.data is not None and hdu.data.ndim == 1:
            data = hdu.data.astype(np.float64)
            header = hdu.header
            break

    if data is None:
        return None, None, None  # 1차원 스펙트럼 데이터가 없는 경우

    n = data.size

    # WCS 정보로 파장축 생성 시도
    try:
        w = WCS(header, naxis=1)
        pix = np.arange(n)
        wavelength = w.wcs_pix2world(pix, 0)[0]
    except Exception:
        crval1 = header.get("CRVAL1", 1)
        cdelt1 = header.get("CDELT1", header.get("CD1_1", 1))
        crpix1 = header.get("CRPIX1", 1)
        pix = np.arange(n)
        wavelength = crval1 + (pix + 1 - crpix1) * cdelt1

    # 단위를 미터로 변환 (기본값: Angstrom으로 가정)
    unit = str(header.get("CUNIT1", "Angstrom")).lower()
    if "nm" in unit:
        wavelength_m = wavelength * 1e-9
    elif "angstrom" in unit or unit == "a":
        wavelength_m = wavelength * 1e-10
    elif "um" in unit or "micron" in unit:
        wavelength_m = wavelength * 1e-6
    else:
        # 단위 정보가 없으면 Angstrom으로 가정
        wavelength_m = wavelength * 1e-10

    flux = data
    return wavelength_m, flux, header


# ------------------------------------------------------------------
# 3. 흑체 온도 피팅
# ------------------------------------------------------------------
def fit_temperature(wavelength_m, flux):
    """
    scipy.optimize.curve_fit으로 Planck 함수를 스펙트럼에 피팅하여
    온도(T)와 오차를 반환.
    """
    # 음수/0 제거 및 정규화
    mask = (wavelength_m > 0) & np.isfinite(flux)
    wavelength_m = wavelength_m[mask]
    flux = flux[mask]

    flux_norm = flux / np.max(flux)

    T0 = wien_peak_temperature(wavelength_m, flux)

    def model(wl, T, scale):
        planck = planck_lambda(wl, T)
        return scale * planck / np.max(planck)

    popt, pcov = curve_fit(
        model, wavelength_m, flux_norm,
        p0=[T0, 1.0],
        bounds=([500, 0], [50000, 10]),
        maxfev=10000,
    )
    T_fit, scale_fit = popt
    T_err = np.sqrt(pcov[0, 0])
    return T_fit, T_err, model, wavelength_m, flux_norm


# ------------------------------------------------------------------
# 4. WCS를 이용한 천체 위치(RA/Dec) 계산
# ------------------------------------------------------------------
def get_sky_position(hdul):
    """
    2차원 이미지 HDU에서 WCS 정보를 읽어 천체(가장 밝은 픽셀 또는 이미지 중심)의
    적경(RA), 적위(Dec)를 계산.
    """
    for hdu in hdul:
        if hdu.data is not None and hdu.data.ndim == 2:
            data = hdu.data
            header = hdu.header
            try:
                w = WCS(header)
            except Exception:
                return None, None, None

            # 가장 밝은 픽셀(천체로 추정)을 대상 좌표로 사용
            y_max, x_max = np.unravel_index(np.nanargmax(data), data.shape)
            ra, dec = w.wcs_pix2world(x_max, y_max, 0)

            sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            return sky_coord, (x_max, y_max), data
    return None, None, None


# ------------------------------------------------------------------
# 5. Streamlit UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="천체 위치·온도 분석", layout="wide")
    st.title("🔭 천체 FITS 분석: 위치(WCS) & 온도(스펙트럼) 계산")

    st.markdown(
        """
        FITS 파일을 업로드하면:
        1. **2차원 이미지 데이터**가 있으면 WCS 정보로 천체의 하늘 좌표(RA/Dec)를 계산합니다.
        2. **1차원 스펙트럼 데이터**가 있으면 흑체복사(Planck) 피팅으로 표면 온도를 추정합니다.
        """
    )

    uploaded_file = st.file_uploader("FITS 파일 업로드 (.fits, .fit)", type=["fits", "fit"])

    if uploaded_file is None:
        st.info("분석할 FITS 파일을 업로드해주세요.")
        return

    try:
        file_bytes = uploaded_file.read()
        hdul = fits.open(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"FITS 파일을 여는 데 실패했습니다: {e}")
        return

    col1, col2 = st.columns(2)

    # ---------------- 위치(WCS) ----------------
    with col1:
        st.subheader("📍 천체 위치 (WCS)")
        sky_coord, pixel_pos, image_data = get_sky_position(hdul)
        if sky_coord is not None:
            st.write(f"**RA (적경):** {sky_coord.ra.to_string(unit=u.hourangle, sep=':')} "
                     f"({sky_coord.ra.deg:.6f}°)")
            st.write(f"**Dec (적위):** {sky_coord.dec.to_string(unit=u.deg, sep=':')} "
                     f"({sky_coord.dec.deg:.6f}°)")

            fig, ax = plt.subplots()
            vmin, vmax = np.nanpercentile(image_data, [5, 99])
            ax.imshow(image_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.scatter(pixel_pos[0], pixel_pos[1], s=120, facecolors="none",
                       edgecolors="red", linewidths=2)
            ax.set_title("Detected Object Position (red circle)")
            ax.set_xlabel("X (pixel)")
            ax.set_ylabel("Y (pixel)")
            st.pyplot(fig)
        else:
            st.warning("이미지(2D) 데이터 또는 WCS 정보를 찾지 못했습니다.")

    # ---------------- 온도(스펙트럼) ----------------
    with col2:
        st.subheader("🌡️ 온도 (흑체복사 스펙트럼 피팅)")
        wavelength_m, flux, spec_header = extract_spectrum(hdul)
        if wavelength_m is not None:
            try:
                T_fit, T_err, model, wl_used, flux_norm = fit_temperature(wavelength_m, flux)

                st.metric("추정 온도", f"{T_fit:,.0f} K", delta=f"± {T_err:,.0f} K")

                fig2, ax2 = plt.subplots()
                wl_nm = wl_used * 1e9  # nm 단위로 표시
                ax2.plot(wl_nm, flux_norm, "o", ms=3, alpha=0.6, label="Observed spectrum")

                wl_smooth = np.linspace(wl_used.min(), wl_used.max(), 500)
                ax2.plot(wl_smooth * 1e9, model(wl_smooth, T_fit, 1.0), "-", lw=2,
                         label=f"Blackbody fit (T={T_fit:,.0f} K)")

                ax2.set_xlabel("Wavelength (nm)")
                ax2.set_ylabel("Normalized flux")
                ax2.legend()
                ax2.set_title("Spectrum & Blackbody Fit")
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"온도 피팅에 실패했습니다: {e}")
        else:
            st.warning("1차원 스펙트럼(flux-wavelength) 데이터를 찾지 못했습니다.")

    with st.expander("FITS 헤더 보기"):
        st.text(repr(hdul[0].header))


if __name__ == "__main__":
    main()
