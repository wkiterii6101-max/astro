"""
FITS WCS 천체 위치 계산기
--------------------------------
FITS 이미지를 업로드하면 헤더의 WCS(World Coordinate System) 정보를 이용해
픽셀 좌표 <-> 하늘 좌표(RA/Dec)를 상호 변환해주는 Streamlit 앱입니다.

실행:
    streamlit run streamlit_app.py

필요 패키지는 requirements.txt 참고
"""

import io

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ZScaleInterval, ImageNormalize
import warnings

# streamlit-image-coordinates: 이미지를 클릭해서 픽셀 좌표를 얻기 위한 컴포넌트
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

warnings.simplefilter("ignore", category=FITSFixedWarning)

st.set_page_config(page_title="FITS WCS 천체 위치 계산기", layout="wide")

st.title("🔭 FITS WCS 천체 위치 계산기")
st.caption("FITS 이미지를 업로드하면 WCS 정보를 이용해 픽셀 좌표를 RA/Dec로 변환합니다.")


# ---------------------------------------------------------------------------
# 유틸 함수
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_fits(file_bytes: bytes):
    """업로드된 바이트에서 FITS를 읽어 (data, header, hdu 인덱스) 반환"""
    hdul = fits.open(io.BytesIO(file_bytes))
    # 이미지 데이터가 들어있는 첫 HDU 탐색 (data가 None이 아닌 것)
    idx = 0
    for i, hdu in enumerate(hdul):
        if hdu.data is not None:
            idx = i
            break
    data = hdul[idx].data
    header = hdul[idx].header
    hdul.close()
    return data, header, idx


def get_wcs(header):
    try:
        w = WCS(header)
        if w.has_celestial:
            return w
        return None
    except Exception as e:
        st.error(f"WCS 파싱 실패: {e}")
        return None


def make_display_image(data: np.ndarray):
    """2D 이미지 데이터를 ZScale로 정규화하여 matplotlib figure 생성"""
    # 다차원(예: 3D 큐브)일 경우 첫 프레임만 사용
    arr = np.array(data, dtype=float)
    while arr.ndim > 2:
        arr = arr[0]

    norm = ImageNormalize(arr, interval=ZScaleInterval())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, origin="lower", cmap="gray", norm=norm)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    fig.tight_layout()
    return fig, arr


def pixel_to_radec(w: WCS, x: float, y: float):
    sky = w.pixel_to_world(x, y)
    if isinstance(sky, SkyCoord):
        return sky
    # 일부 WCS(3D 등)는 리스트를 반환할 수 있음
    return sky[0]


def radec_to_pixel(w: WCS, ra_deg: float, dec_deg: float):
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = w.world_to_pixel(sky)
    return float(x), float(y)


# ---------------------------------------------------------------------------
# 메인 UI
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "FITS 파일 업로드 (.fits, .fit, .fts)",
    type=["fits", "fit", "fts"],
)

if uploaded_file is None:
    st.info("좌측 상단에서 FITS 파일을 업로드해 주세요.")
    st.stop()

file_bytes = uploaded_file.read()

with st.spinner("FITS 파일을 읽는 중..."):
    try:
        data, header, hdu_idx = load_fits(file_bytes)
    except Exception as e:
        st.error(f"FITS 파일을 읽을 수 없습니다: {e}")
        st.stop()

if data is None:
    st.error("이미지 데이터를 찾을 수 없습니다. (선택된 HDU에 데이터 없음)")
    st.stop()

w = get_wcs(header)

col_img, col_info = st.columns([2, 1])

with col_info:
    st.subheader("📋 헤더 / WCS 정보")
    st.write(f"**사용된 HDU 인덱스:** {hdu_idx}")
    st.write(f"**이미지 크기 (shape):** {data.shape}")

    if w is None:
        st.warning("이 FITS 헤더에는 유효한 천체 좌표계(WCS) 정보가 없습니다. "
                   "CRVAL/CRPIX/CDELT/CTYPE 등의 키워드를 확인해 주세요.")
    else:
        st.success("WCS 정보를 정상적으로 읽었습니다.")
        wcs_keys = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
                    "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CD1_1", "CD2_2"]
        found = {k: header[k] for k in wcs_keys if k in header}
        if found:
            st.json(found)

    with st.expander("전체 FITS 헤더 보기"):
        st.text(repr(header))

with col_img:
    st.subheader("🖼️ 이미지")
    fig, arr = make_display_image(data)

    if w is not None and HAS_CLICK:
        st.caption("이미지를 클릭하면 해당 픽셀의 RA/Dec를 계산합니다.")
        # matplotlib figure -> PNG로 변환 후 클릭 좌표 컴포넌트에 표시
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        from PIL import Image
        pil_img = Image.open(buf)

        coords = streamlit_image_coordinates(pil_img, key="fits_click")
        plt.close(fig)

        if coords is not None:
            # 화면상 픽셀 좌표를 원본 이미지 픽셀 좌표로 환산
            disp_w, disp_h = pil_img.size
            data_h, data_w = arr.shape
            click_x_disp, click_y_disp = coords["x"], coords["y"]

            # matplotlib figure의 여백을 고려하기보다는 근사 비율 변환 사용
            px = click_x_disp / disp_w * data_w
            # imshow origin='lower' 이므로 화면 y(위->아래)를 데이터 y(아래->위)로 반전
            py = data_h - (click_y_disp / disp_h * data_h)

            st.session_state["pick_x"] = px
            st.session_state["pick_y"] = py
    else:
        st.pyplot(fig)
        plt.close(fig)
        if w is not None and not HAS_CLICK:
            st.caption(
                "💡 이미지를 클릭해서 좌표를 얻으려면 "
                "`pip install streamlit-image-coordinates` 를 설치하세요. "
                "지금은 아래에서 픽셀 좌표를 직접 입력해 주세요."
            )

# ---------------------------------------------------------------------------
# 좌표 변환 UI
# ---------------------------------------------------------------------------

if w is not None:
    st.divider()
    st.subheader("📐 좌표 변환")

    tab_pix2sky, tab_sky2pix = st.tabs(["픽셀 → 하늘좌표 (RA/Dec)", "하늘좌표 → 픽셀"])

    ny, nx = arr.shape

    with tab_pix2sky:
        c1, c2, c3 = st.columns([1, 1, 1])
        default_x = st.session_state.get("pick_x", nx / 2)
        default_y = st.session_state.get("pick_y", ny / 2)

        x_pix = c1.number_input("X (pixel)", value=float(default_x), format="%.3f")
        y_pix = c2.number_input("Y (pixel)", value=float(default_y), format="%.3f")

        if c3.button("계산하기", type="primary", key="btn_pix2sky"):
            try:
                sky = pixel_to_radec(w, x_pix, y_pix)
                ra_deg = sky.ra.deg
                dec_deg = sky.dec.deg

                r1, r2 = st.columns(2)
                with r1:
                    st.metric("RA (deg)", f"{ra_deg:.6f}")
                    st.metric("RA (hms)", sky.ra.to_string(unit=u.hourangle, sep=":", precision=2))
                with r2:
                    st.metric("Dec (deg)", f"{dec_deg:.6f}")
                    st.metric("Dec (dms)", sky.dec.to_string(unit=u.deg, sep=":", precision=2))

                st.code(
                    f"RA  = {ra_deg:.6f} deg  ({sky.ra.to_string(unit=u.hourangle, sep=':', precision=2)})\n"
                    f"Dec = {dec_deg:.6f} deg ({sky.dec.to_string(unit=u.deg, sep=':', precision=2)})",
                    language="text",
                )
            except Exception as e:
                st.error(f"좌표 변환 중 오류가 발생했습니다: {e}")

    with tab_sky2pix:
        c1, c2, c3 = st.columns([1, 1, 1])
        ra_in = c1.number_input("RA (deg)", value=0.0, format="%.6f")
        dec_in = c2.number_input("Dec (deg)", value=0.0, format="%.6f")

        if c3.button("계산하기", type="primary", key="btn_sky2pix"):
            try:
                x_out, y_out = radec_to_pixel(w, ra_in, dec_in)
                r1, r2 = st.columns(2)
                r1.metric("X (pixel)", f"{x_out:.3f}")
                r2.metric("Y (pixel)", f"{y_out:.3f}")

                in_bounds = (0 <= x_out < nx) and (0 <= y_out < ny)
                if not in_bounds:
                    st.warning("계산된 픽셀 좌표가 이미지 범위를 벗어났습니다 (이미지 밖의 천체일 수 있습니다).")
            except Exception as e:
                st.error(f"좌표 변환 중 오류가 발생했습니다: {e}")

st.divider()
st.caption(
    "Made with Astropy WCS + Streamlit. "
    "이미지 좌상단/좌하단 원점 규약은 FITS 표준(origin=0, 좌하단)을 따릅니다."
)
