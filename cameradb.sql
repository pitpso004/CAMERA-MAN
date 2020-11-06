-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Nov 06, 2020 at 08:13 AM
-- Server version: 10.4.13-MariaDB
-- PHP Version: 7.4.8

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `cameradb`
--

-- --------------------------------------------------------

--
-- Table structure for table `alerts_member`
--

CREATE TABLE `alerts_member` (
  `alert_id` int(11) NOT NULL,
  `alert_name` varchar(20) NOT NULL,
  `alert_date` varchar(8) NOT NULL,
  `alert_time` varchar(8) NOT NULL,
  `member_email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `images_member`
--

CREATE TABLE `images_member` (
  `image_id` int(11) NOT NULL,
  `image_folder` varchar(50) NOT NULL,
  `image_count` int(11) NOT NULL,
  `member_email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `linetoken_member`
--

CREATE TABLE `linetoken_member` (
  `token_id` int(11) NOT NULL,
  `token_name` varchar(50) NOT NULL,
  `token_line` varchar(43) NOT NULL,
  `member_email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `register_member`
--

CREATE TABLE `register_member` (
  `member_id` int(11) NOT NULL,
  `member_name` varchar(50) NOT NULL,
  `member_email` varchar(50) NOT NULL,
  `member_password` varchar(50) NOT NULL,
  `member_state` varchar(1) NOT NULL,
  `token_id` int(11) NOT NULL,
  `delay_alert` int(1) NOT NULL,
  `delay_record` int(4) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `videos_member`
--

CREATE TABLE `videos_member` (
  `videos_id` int(11) NOT NULL,
  `videos_name` varchar(20) NOT NULL,
  `videos_date` varchar(8) NOT NULL,
  `videos_timeStart` varchar(8) NOT NULL,
  `videos_timeEnd` varchar(8) NOT NULL,
  `member_email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `alerts_member`
--
ALTER TABLE `alerts_member`
  ADD PRIMARY KEY (`alert_id`),
  ADD UNIQUE KEY `alert_name` (`alert_name`);

--
-- Indexes for table `images_member`
--
ALTER TABLE `images_member`
  ADD PRIMARY KEY (`image_id`);

--
-- Indexes for table `linetoken_member`
--
ALTER TABLE `linetoken_member`
  ADD PRIMARY KEY (`token_id`);

--
-- Indexes for table `register_member`
--
ALTER TABLE `register_member`
  ADD PRIMARY KEY (`member_id`);

--
-- Indexes for table `videos_member`
--
ALTER TABLE `videos_member`
  ADD PRIMARY KEY (`videos_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `alerts_member`
--
ALTER TABLE `alerts_member`
  MODIFY `alert_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1108;

--
-- AUTO_INCREMENT for table `images_member`
--
ALTER TABLE `images_member`
  MODIFY `image_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=984;

--
-- AUTO_INCREMENT for table `linetoken_member`
--
ALTER TABLE `linetoken_member`
  MODIFY `token_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=22;

--
-- AUTO_INCREMENT for table `register_member`
--
ALTER TABLE `register_member`
  MODIFY `member_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `videos_member`
--
ALTER TABLE `videos_member`
  MODIFY `videos_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=64;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
