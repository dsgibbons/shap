import matplotlib.pyplot as plt
import pytest
import shap


@pytest.mark.mpl_image_compare
def test_violin(explainer): # pylint: disable=redefined-outer-name
    """ Make sure the violin plot is unchanged.
    """
    fig = plt.figure()
    shap_values = explainer.shap_values(explainer.data)
    shap.plots.violin(shap_values)
    plt.tight_layout()
    return fig
