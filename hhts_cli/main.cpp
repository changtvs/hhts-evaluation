// Copyright (c) Technische Hochschule Nürnberg, Game Tech Lab.
// All rights reserved.

#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <bitset>
#include "io_util.h"
#include "superpixel_tools.h"
#include "visualization.h"
#include "hhts.h"

using namespace cv;

/** \brief Command line tool for running W.
 * Usage:
 * \code{sh}
 *   $ ../bin/hts_cli --help
 *   Allowed options:
 *     -h [ --help ]                   produce help message
 *     -i [ --input ] arg              the folder to process (can also be passed as
 *                                     positional argument)
 *     --smax arg (=0)
 *     --smin arg (=0)
 *     --rgb (=true)
 *     --hsv (=true)
 *     --lab (=true)
 *     --stddmin (=0.0)
 *     --histwmin (=5)
 *     --thresholds (=1)
 *     --bins (=16)
 *     --cw (=true)
 *     --nomerge (=false)
 *     -o [ --csv ] arg                specify the output directory (default is
 *                                     ./output)
 *     -v [ --vis ] arg                visualize contours
 *     -x [ --prefix ] arg             output file prefix
 *     -w [ --wordy ]                  verbose/wordy/debug
 * \endcode
 * \author David Stutz
 */
int main(int argc, const char **argv)
{

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message")
    ("input,i", boost::program_options::value<std::string>(), "the folder to process (can also be passed as positional argument)")
    ("superpixels,s", boost::program_options::value<vector<int>>()->multitoken(), "numbers of superpixels")
    ("splitThreshold,t", boost::program_options::value<double>()->default_value(0.0), "min stddev * histWidth of superpixles")
    ("nrgb", "do not use rgb channel")
    ("nhsv", "do not use hsv channel")
    ("nlab", "do not use lab channel")
    ("blur", "apply blur to channels")
    ("bins,b", boost::program_options::value<int>()->default_value(32), "number of histogram bins")
    ("minSize,m", boost::program_options::value<int>()->default_value(64), "minimum size of segments")
    ("csv,o", boost::program_options::value<std::string>()->default_value(""), "specify the output directory (default is ./output)")
    ("vis,v", boost::program_options::value<std::string>()->default_value(""), "visualize contours")
    ("prefix,x", boost::program_options::value<std::string>()->default_value(""), "output file prefix")
    ("wordy,w", "verbose/wordy/debug");

    boost::program_options::positional_options_description positionals;
    positionals.add("input", 1);

    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end())
    {
        std::cout << desc << std::endl;
        return 1;
    }

    boost::filesystem::path output_dir(parameters["csv"].as<std::string>());
    if (!output_dir.empty())
    {
        if (!boost::filesystem::is_directory(output_dir))
        {
            boost::filesystem::create_directories(output_dir);
        }
    }

    boost::filesystem::path vis_dir(parameters["vis"].as<std::string>());
    if (!vis_dir.empty())
    {
        if (!boost::filesystem::is_directory(vis_dir))
        {
            boost::filesystem::create_directories(vis_dir);
        }
    }

    boost::filesystem::path input_dir(parameters["input"].as<std::string>());
    if (!boost::filesystem::is_directory(input_dir))
    {
        std::cout << "Image directory not found ..." << std::endl;
        return 1;
    }

    std::string prefix = parameters["prefix"].as<std::string>();

    bool wordy = false;
    if (parameters.find("wordy") != parameters.end())
    {
        wordy = true;
    }

    vector<int> superpixels{};
    if (parameters.find("superpixels") != parameters.end())
    {
        superpixels = parameters["superpixels"].as<vector<int>>();
        for (int sp : superpixels)
        {
            boost::filesystem::path spPath = output_dir / to_string(sp);
            if (!boost::filesystem::is_directory(spPath))
            {
                boost::filesystem::create_directories(spPath);
            }
        }
    }
    double splitThreshold = 0.0;
    if (parameters.find("splitThreshold") != parameters.end())
    {
        splitThreshold = parameters["splitThreshold"].as<double>();
    }
    bool rgb = true, hsv = true, lab = true;
    if (parameters.find("nrgb") != parameters.end())
    {
        rgb = false;
    }
    if (parameters.find("nhsv") != parameters.end())
    {
        hsv = false;
    }
    if (parameters.find("nlab") != parameters.end())
    {
        lab = false;
    }
    bool applyBlur = false;
    if (parameters.find("blur") != parameters.end())
    {
        applyBlur = true;
    }
    int bins = 32;
    if (parameters.find("bins") != parameters.end())
    {
        bins = parameters["bins"].as<int>();
    }
    int minSegmentSize = 64;
    if (parameters.find("minSize") != parameters.end())
    {
        minSegmentSize = parameters["minSize"].as<int>();
    }

    std::multimap<std::string, boost::filesystem::path> images;
    std::vector<std::string> extensions;
    IOUtil::getImageExtensions(extensions);
    IOUtil::readDirectory(input_dir, extensions, images);

    double totalWall = 0;
    double total = 0;
    int count = 0;
    for (std::multimap<std::string, boost::filesystem::path>::iterator it = images.begin();
         it != images.end(); ++it)
    {

        cv::Mat image = cv::imread(it->first);

        // set params
        vector<Mat> labels;
        vector<int> labelCounts;
        int colorChannels = 0;
        if (rgb)
        {
            colorChannels |= HHTS::ColorChannel::RGB;
        }
        if (hsv)
        {
            colorChannels |= HHTS::ColorChannel::HSV;
        }
        if (lab)
        {
            colorChannels |= HHTS::ColorChannel::LAB;
        }

        boost::timer::cpu_timer timer;
        labelCounts = HHTS::hhts(image, labels, superpixels, splitThreshold, bins, minSegmentSize, colorChannels, applyBlur, noArray());

        boost::chrono::duration<double> secondsWall = boost::chrono::nanoseconds(timer.elapsed().wall);
        boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(timer.elapsed().user + timer.elapsed().system);
        double elapsedWall = secondsWall.count();
        totalWall += elapsedWall;
        double elapsed = seconds.count();
        total += elapsed;
        count++;

        for (int i = 0; i < labels.size(); ++i)
        {
            int unconnected_components = SuperpixelTools::relabelConnectedSuperpixels(labels[i]);
            // cout << labelCounts[i] << ": " << unconnected_components << " unconnected" << endl;
        }

        if (wordy)
        {
            // cout << count << "/" << images.size() << ": " << SuperpixelTools::countSuperpixels(labels) << " superpixels (" << elapsed << "/" << (total / count) << " - " << elapsedWall << "/" << (totalWall / count) << ") " << unconnected_components << " not connected" << endl;

            // std::cout << SuperpixelTools::countSuperpixels(labels) << " superpixels for " << it->first
            //           << " (" << unconnected_components << " not connected; "
            //           << elapsed << ")." << std::endl;
        }

        for (int i = 0; i < labels.size(); ++i)
        {
            if (!output_dir.empty())
            {
                boost::filesystem::path csv_file(output_dir / to_string(superpixels[i]) / boost::filesystem::path(prefix + it->second.stem().string() + ".csv"));
                IOUtil::writeMatCSV<int>(csv_file, labels[i]);
            }

            if (!vis_dir.empty())
            {
                boost::filesystem::path contours_file(vis_dir / to_string(superpixels[i]) / boost::filesystem::path(prefix + it->second.stem().string() + ".png"));
                cv::Mat image_contours;
                Visualization::drawContours(image, labels[i], image_contours);
                cv::imwrite(contours_file.string(), image_contours);
            }
        }
    }

    if (wordy)
    {
        std::cout << "Average time: " << total / count << " - " << totalWall / count << "." << std::endl;
    }

    if (!output_dir.empty())
    {
        std::ofstream runtime_file(output_dir.string() + "/" + prefix + "runtime.txt",
                                   std::ofstream::out | std::ofstream::app);

        runtime_file << total / count << " " << totalWall / count << "\n";
        runtime_file.close();
    }

    return 0;
}
