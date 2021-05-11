/**
********************************************************************************
* @file     apps/p4/Base/src/main.c
* @author   Zephyr Development Team, modified by Rhys Sneddon - 44785954
* @date     11052021
* @brief    main.c file for base node in Prac 4
********************************************************************************
*/

#include <zephyr/types.h>
#include <stddef.h>
#include <errno.h>
#include <zephyr.h>
#include <sys/printk.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/hci.h>
#include <bluetooth/conn.h>
#include <bluetooth/uuid.h>
#include <bluetooth/gatt.h>
#include <sys/byteorder.h>
#include <zephyr.h>
#include <sys/printk.h>
#include <sys/util.h>
#include <string.h>
#include <usb/usb_device.h>
#include <drivers/uart.h>

static void start_scan(void);

static struct bt_conn *default_conn;

volatile char str1[100]; 
volatile char str2[100];

static void device_found(const bt_addr_le_t *addr, int8_t rssi, uint8_t type,
			 struct net_buf_simple *ad) {

	char addr_str[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(addr, addr_str, sizeof(addr_str));

	if (   (ad->data[2] == 0x63) 
		&& (ad->data[3] == 0x73) 
		&& (ad->data[4] == 0x00) 
		&& (ad->data[5] == 0x00)) {
	
		// for (int i=0; i<ad->len; i+=1){
		// 	printk("%02X ", ad->data[i]);
		// }

		// output json to serial
		sprintf(str1, "[[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],1]\n", 
		(int8_t)ad->data[6], (uint8_t)ad->data[7], 
		(int8_t)ad->data[8], (uint8_t)ad->data[9], 
		(int8_t)ad->data[10], (uint8_t)ad->data[11],
		(int8_t)ad->data[12], (uint8_t)ad->data[13],

		(int8_t)ad->data[14], (uint8_t)ad->data[15], 
		(int8_t)ad->data[16], (uint8_t)ad->data[17], 
		(int8_t)ad->data[18], (uint8_t)ad->data[19],
		(int8_t)ad->data[20], (uint8_t)ad->data[21]);
	}

	if (   (ad->data[2] == 0x63) 
		&& (ad->data[3] == 0x73) 
		&& (ad->data[4] == 0x00) 
		&& (ad->data[5] == 0x09)) {
	
		// for (int i=0; i<ad->len; i+=1){
		// 	printk("%02X ", ad->data[i]);
		// }

		// output json to serial
		sprintf(str2, "[[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],[%d,%u],2]\n", 
		(int8_t)ad->data[6], (uint8_t)ad->data[7], 
		(int8_t)ad->data[8], (uint8_t)ad->data[9], 
		(int8_t)ad->data[10], (uint8_t)ad->data[11],
		(int8_t)ad->data[12], (uint8_t)ad->data[13],

		(int8_t)ad->data[14], (uint8_t)ad->data[15], 
		(int8_t)ad->data[16], (uint8_t)ad->data[17], 
		(int8_t)ad->data[18], (uint8_t)ad->data[19],
		(int8_t)ad->data[20], (uint8_t)ad->data[21]);
	}


	/* connect only to devices in close proximity */
	if (rssi < -70) {
		return;
	}

	if (bt_le_scan_stop()) {
		return;
	}
}

int printString = 1;

static void start_scan(void)
{
	int err;

	/* This demo doesn't require active scan */
	err = bt_le_scan_start(BT_LE_SCAN_PASSIVE, device_found);
	if (err) {
		return;
	} else {
		if (printString == 1) {
			printk("%s\n", str1);
		} else if (printString == 2) {
			printk("%s\n", str2);
		}
	}

	//printk("Scanning successfully started\n");
}

static void connected(struct bt_conn *conn, uint8_t err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (err) {
		printk("Failed to connect to %s (%u)\n", addr, err);

		bt_conn_unref(default_conn);
		default_conn = NULL;

		start_scan();
		return;
	}

	if (conn != default_conn) {
		return;
	}

	printk("Connected: %s\n", addr);

	bt_conn_disconnect(conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];

	if (conn != default_conn) {
		return;
	}

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	printk("Disconnected: %s (reason 0x%02x)\n", addr, reason);

	bt_conn_unref(default_conn);
	default_conn = NULL;

	start_scan();
}

static struct bt_conn_cb conn_callbacks = {
		.connected = connected,
		.disconnected = disconnected,
};

void main(void)
{

	const struct device *dev = device_get_binding(
		CONFIG_UART_CONSOLE_ON_DEV_NAME);
	uint32_t dtr = 0;

	if (usb_enable(NULL)) {
		return;
	}

	/* Poll if the DTR flag was set, optional */
	while (!dtr) {
		uart_line_ctrl_get(dev, UART_LINE_CTRL_DTR, &dtr);
	}

	if (strlen(CONFIG_UART_CONSOLE_ON_DEV_NAME) !=
	    strlen("CDC_ACM_0") ||
	    strncmp(CONFIG_UART_CONSOLE_ON_DEV_NAME, "CDC_ACM_0",
		    strlen(CONFIG_UART_CONSOLE_ON_DEV_NAME))) {
		//printk("Error: Console device name is not USB ACM\n");

		return;
	}

	int err;

	err = bt_enable(NULL);
	if (err) {
		//printk("Bluetooth init failed (err %d)\n", err);
		return;
	}

	//printk("Bluetooth initialized\n");

	bt_conn_cb_register(&conn_callbacks);

	start_scan();
}
