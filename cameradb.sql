-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Oct 03, 2020 at 05:49 PM
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

--
-- Dumping data for table `alerts_member`
--

INSERT INTO `alerts_member` (`alert_id`, `alert_name`, `alert_date`, `alert_time`, `member_email`) VALUES
(983, '03102020 152133.jpg', '03/10/20', '15:21:38', 'pitpso005@gmail.com'),
(984, '03102020 152312.jpg', '03/10/20', '15:23:17', 'pitpso005@gmail.com'),
(996, '03102020 211202.jpg', '03/10/20', '21:12:02', 'pitpso005@gmail.com');

-- --------------------------------------------------------

--
-- Table structure for table `images_member`
--

CREATE TABLE `images_member` (
  `image_id` int(11) NOT NULL,
  `image_folder` varchar(50) NOT NULL,
  `member_email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `images_member`
--

INSERT INTO `images_member` (`image_id`, `image_folder`, `member_email`) VALUES
(916, 'non', 'pitpso005@gmail.com'),
(917, 'aof', 'pitpso005@gmail.com'),
(918, 'nack', 'pitpso005@gmail.com');

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

--
-- Dumping data for table `linetoken_member`
--

INSERT INTO `linetoken_member` (`token_id`, `token_name`, `token_line`, `member_email`) VALUES
(5, 'Camera Man', 'N6l8F3kOu3PczpQp1rLy3vTF4OL7OYSxfnj1jFLpFQO', 'pitpso005@gmail.com'),
(6, 'ตัวเอง', '7NuQEfNJ3j74mnUVEBEturX11KZ7xPkrqL7o7iGpEAu', 'pitpso005@gmail.com'),
(7, 'เจมส์', 'ZOerDErhG8KP4ENzGBUgEtBQML7IdaTSopotvHxgFEv', 'pitpso005@gmail.com');

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

--
-- Dumping data for table `register_member`
--

INSERT INTO `register_member` (`member_id`, `member_name`, `member_email`, `member_password`, `member_state`, `token_id`, `delay_alert`, `delay_record`) VALUES
(1, 'non', 'pitpso005@gmail.com', '1234', '0', 6, 5, 30),
(4, 'ohm', 'ommi406@outlook.com', '1234', '0', 0, 5, 30);

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
-- Dumping data for table `videos_member`
--

INSERT INTO `videos_member` (`videos_id`, `videos_name`, `videos_date`, `videos_timeStart`, `videos_timeEnd`, `member_email`) VALUES
(40, '03102020 152317.mp4', '03/10/20', '15:23:18', '15:23:48', 'pitpso005@gmail.com'),
(44, '03102020 155943.mp4', '03/10/20', '15:59:43', '16:00:14', 'pitpso005@gmail.com'),
(45, '03102020 211459.mp4', '03/10/20', '21:14:59', '21:15:30', 'pitpso005@gmail.com');

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
  MODIFY `alert_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1025;

--
-- AUTO_INCREMENT for table `images_member`
--
ALTER TABLE `images_member`
  MODIFY `image_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=921;

--
-- AUTO_INCREMENT for table `linetoken_member`
--
ALTER TABLE `linetoken_member`
  MODIFY `token_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `register_member`
--
ALTER TABLE `register_member`
  MODIFY `member_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT for table `videos_member`
--
ALTER TABLE `videos_member`
  MODIFY `videos_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=49;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
