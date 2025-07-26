from setuptools import setup

package_name = "lane_follower_pkg"

setup(
    name=package_name,
    version="0.0.1",
    packages=[
        package_name,
        f"{package_name}.model",
    ],
    package_dir={"": "src"},
    install_requires=[
        "setuptools",
        "torch",
        "opencv-python-headless",
        "albumentations",
        "numpy",
        "opencv-python",
        "cv_bridge",
    ],
    zip_safe=True,
    author="Your Name",
    author_email="you@example.com",
    description="Lane following node using U-Net on Jetson/ROS2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lane_follower = lane_follower_pkg.lane_follower_node:main",
        ],
    },
)
